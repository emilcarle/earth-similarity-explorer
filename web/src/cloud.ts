import type { Layer } from "@deck.gl/core";
import { LayerExtension } from "@deck.gl/core";
import type { Device, Texture } from "@luma.gl/core";
import type { ShaderModule } from "@luma.gl/shadertools";
import type { Cloud } from "./api";
import {
  AREA_COLOR,
  CLOUD_JITTER,
  HIGHLIGHT_COLOR,
  POSITION_BYTE_CENTRE,
  POSITION_BYTE_HALF_RANGE,
  QUERY_COLOR,
  SCORE_BYTE_MAX,
  SEGMENT_TEXTURE_WIDTH,
} from "./settings";

// This extension adds GPU code to a PointCloudLayer of pixels. It moves each point
// between its map position and its segment's cloud position, and colours it by its
// segment's score in the latest search, like the map.

/** The extra props a layer with this extension takes. */
export type CloudExtensionProps = {
  // Three position bytes and a selected flag per segment, from createPositionTexture
  positionTexture: Texture | null;
  // One score byte per segment, from createScoreTexture
  scoreTexture: Texture | null;
  // Score byte at or above which a segment is highlighted
  threshold: number;
  // The clicked segment, or −1 before the first click
  queryId: number;
  // 0 places the points on the map, 1 in the cloud
  morph: number;
  // Half the map's width in scene metres, which the cloud's edge bytes reach
  halfWidth: number;
};

type Vec4 = [number, number, number, number];
// Values the GPU code reads. Uniforms are single values, bindings are textures.
type CloudUniforms = {
  areaColor: Vec4;
  highlightColor: Vec4;
  queryColor: Vec4;
  threshold: number;
  queryId: number;
  morph: number;
  halfWidth: number;
};
type CloudBindings = {
  cloud_positionTexture: Texture;
  cloud_scoreTexture: Texture;
};

// GPU code (GLSL) shared by the whole vertex shader. The order of the values in the
// uniform block must match uniformTypes below.
const vs = /* glsl */ `\
layout(std140) uniform cloudUniforms {
  vec4 areaColor;
  vec4 highlightColor;
  vec4 queryColor;
  float threshold;
  float queryId;
  float morph;
  float halfWidth;
} cloud;

uniform sampler2D cloud_positionTexture;
uniform sampler2D cloud_scoreTexture;

// Per-point values in addition to the layer's position
in float instanceSegmentIds;
in vec3 instanceJitters;

// The texel of a segment, which sits at column id % width and row id / width
ivec2 cloud_texel(int id) {
  return ivec2(id % ${SEGMENT_TEXTURE_WIDTH}, id / ${SEGMENT_TEXTURE_WIDTH});
}
`;

const cloudModule = {
  name: "cloud",
  vs,
  inject: {
    // Runs for every point after deck.gl has placed it on the screen. The point's
    // size is added afterwards, so replacing its position moves the whole point.
    "vs:DECKGL_FILTER_GL_POSITION": /* glsl */ `
      vec4 stored = texelFetch(cloud_positionTexture, cloud_texel(int(instanceSegmentIds)), 0);
      // Bytes 1 to 255 become −1 to 1 around the centre byte, scaled to the map's width,
      // plus the point's random offset
      vec3 bytes = round(stored.xyz * 255.0);
      vec3 cloudPosition = (bytes - ${POSITION_BYTE_CENTRE}.0) / ${POSITION_BYTE_HALF_RANGE}.0 * cloud.halfWidth
        + instanceJitters * ${CLOUD_JITTER} * cloud.halfWidth;
      // Blend between the map position and the cloud position
      vec3 blended = mix(geometry.worldPosition, cloudPosition, cloud.morph);
      position = project_position_to_clipspace(blended, vec3(0.0), vec3(0.0));
      // Outside the map, points of unselected segments (alpha 0) move beyond the far end
      // of the view, where the GPU skips them
      if (cloud.morph > 0.0 && stored.a < 0.5) {
        position = vec4(0.0, 0.0, 2.0, 1.0);
      }
    `,
    // Colour the point like its segment on the map
    "vs:DECKGL_FILTER_COLOR": /* glsl */ `
      int id = int(instanceSegmentIds);
      float score = round(texelFetch(cloud_scoreTexture, cloud_texel(id), 0).r * 255.0);
      if (float(id) == cloud.queryId) {
        color = cloud.queryColor;
      } else if (cloud.queryId >= 0.0 && score >= cloud.threshold) {
        color = cloud.highlightColor;
      } else {
        color = cloud.areaColor;
      }
    `,
  },
  uniformTypes: {
    areaColor: "vec4<f32>",
    highlightColor: "vec4<f32>",
    queryColor: "vec4<f32>",
    threshold: "f32",
    queryId: "f32",
    morph: "f32",
    halfWidth: "f32",
  },
  getUniforms: (props?: Partial<CloudUniforms & CloudBindings>) => props ?? {},
} as const satisfies ShaderModule<
  CloudUniforms & CloudBindings,
  CloudUniforms,
  CloudBindings
>;

/** Turn a colour from 0 to 255 into the GPU's 0 to 1, fully opaque. */
function toVec4([red, green, blue]: [number, number, number]): Vec4 {
  return [red / 255, green / 255, blue / 255, 1];
}

/** Moves the points of a PointCloudLayer between map and cloud, and colours them by score. */
export class CloudExtension extends LayerExtension {
  static extensionName = "CloudExtension";
  static defaultProps = {
    positionTexture: null,
    scoreTexture: null,
    threshold: SCORE_BYTE_MAX + 1,
    queryId: -1,
    morph: 0,
    halfWidth: 1,
  };

  /** Add the cloud module to the layer's shaders. */
  getShaders() {
    return { modules: [cloudModule] };
  }

  /** Add the segment ID and the random offset of each point as attributes. */
  initializeState(this: Layer<CloudExtensionProps>) {
    // With binary data, deck.gl takes them from data.attributes.getSegmentId and
    // data.attributes.getJitter without a JavaScript call per point
    this.getAttributeManager()?.addInstanced({
      instanceSegmentIds: { size: 1, accessor: "getSegmentId" },
      instanceJitters: { size: 3, accessor: "getJitter" },
    });
  }

  /** Hand the textures and values to the GPU before each draw. */
  draw(this: Layer<CloudExtensionProps>) {
    const {
      positionTexture,
      scoreTexture,
      threshold,
      queryId,
      morph,
      halfWidth,
    } = this.props;
    if (!positionTexture || !scoreTexture) {
      return;
    }
    this.setShaderModuleProps({
      cloud: {
        cloud_positionTexture: positionTexture,
        cloud_scoreTexture: scoreTexture,
        threshold,
        queryId,
        morph,
        halfWidth,
        areaColor: toVec4(AREA_COLOR),
        highlightColor: toVec4(HIGHLIGHT_COLOR),
        queryColor: toVec4(QUERY_COLOR),
      },
    });
  }
}

/** Put each selected segment's position bytes into a texture the GPU looks up by segment ID. */
export function createPositionTexture(
  device: Device,
  segmentCount: number,
  cloud: Cloud,
): Texture {
  // Four bytes (x, y, z, selected) per segment in rows of SEGMENT_TEXTURE_WIDTH. All
  // start at 0, so segments outside the selection have alpha 0.
  const height = Math.ceil(segmentCount / SEGMENT_TEXTURE_WIDTH);
  const data = new Uint8Array(SEGMENT_TEXTURE_WIDTH * height * 4);
  // Copy the three bytes of each selected segment to its texel and mark it selected
  cloud.segmentIds.forEach((id, index) => {
    data.set(cloud.positions.subarray(3 * index, 3 * index + 3), 4 * id);
    data[4 * id + 3] = 255;
  });

  // No mipmaps and no blending of neighbours, so each texel stays one segment
  const texture = device.createTexture({
    format: "rgba8unorm",
    width: SEGMENT_TEXTURE_WIDTH,
    height,
    mipLevels: 1,
    sampler: { minFilter: "nearest", magFilter: "nearest" },
  });
  texture.writeData(data);
  return texture;
}
