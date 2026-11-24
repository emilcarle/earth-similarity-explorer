import type { Layer } from "@deck.gl/core";
import { LayerExtension } from "@deck.gl/core";
import type { Device, Texture } from "@luma.gl/core";
import type { ShaderModule } from "@luma.gl/shadertools";
import {
  AREA_COLOR,
  HIGHLIGHT_COLOR,
  NO_SEGMENT_ID,
  QUERY_COLOR,
  SCORE_BYTE_MAX,
  SCORE_TEXTURE_WIDTH,
} from "./settings";

// A layer extension adds GPU code to an existing deck.gl layer. This one makes a BitmapLayer
// of segment IDs draw each pixel by its segment's score in the latest search.

/** The extra props a layer with this extension takes. */
export type SegmentExtensionProps = {
  // One score byte per segment, from createScoreTexture
  scoreTexture: Texture | null;
  // Score byte at or above which a segment is highlighted (SCORE_BYTE_MAX + 1
  // highlights nothing)
  threshold: number;
  // The clicked segment, or −1 before the first click
  queryId: number;
};

type Vec4 = [number, number, number, number];
// Values the GPU code reads. Uniforms are single values, bindings are textures.
type SegmentUniforms = {
  areaColor: Vec4;
  highlightColor: Vec4;
  queryColor: Vec4;
  threshold: number;
  queryId: number;
};
type SegmentBindings = {
  segment_idTexture: Texture;
  segment_scoreTexture: Texture;
};

// GPU code (GLSL) shared by the whole fragment shader. The order of the values in the
// uniform block must match uniformTypes below.
const fs = /* glsl */ `\
layout(std140) uniform segmentUniforms {
  vec4 areaColor;
  vec4 highlightColor;
  vec4 queryColor;
  float threshold;
  float queryId;
} segment;

uniform sampler2D segment_idTexture;
uniform sampler2D segment_scoreTexture;

// Decode the segment ID at a position in the ID raster (0 to 1 in both directions).
// texelFetch reads exactly one pixel, so neighbouring IDs are never blended.
int segment_decodeId(vec2 uv) {
  ivec2 size = textureSize(segment_idTexture, 0);
  ivec2 pixel = clamp(ivec2(uv * vec2(size)), ivec2(0), size - 1);
  ivec3 rgb = ivec3(round(texelFetch(segment_idTexture, pixel, 0).rgb * 255.0));
  // R holds the high byte and B the low byte
  return (rgb.r << 16) | (rgb.g << 8) | rgb.b;
}

// The score byte (0 to 255) of a segment, which sits at column id % width and row id / width
float segment_score(int id) {
  int width = textureSize(segment_scoreTexture, 0).x;
  return round(texelFetch(segment_scoreTexture, ivec2(id % width, id / width), 0).r * 255.0);
}
`;

const segmentModule = {
  name: "segment",
  fs,
  inject: {
    // Runs for every screen pixel just before its colour is written
    "fs:DECKGL_FILTER_COLOR": /* glsl */ `
      int id = segment_decodeId(geometry.uv);
      // Pixels outside the area are not drawn and can't be clicked
      if (id == ${NO_SEGMENT_ID}) {
        discard;
      }
      if (float(id) == segment.queryId) {
        color = segment.queryColor;
      } else if (segment.queryId >= 0.0 && segment_score(id) >= segment.threshold) {
        color = segment.highlightColor;
      } else {
        color = segment.areaColor;
      }
    `,
  },
  uniformTypes: {
    areaColor: "vec4<f32>",
    highlightColor: "vec4<f32>",
    queryColor: "vec4<f32>",
    threshold: "f32",
    queryId: "f32",
  },
  getUniforms: (props?: Partial<SegmentUniforms & SegmentBindings>) =>
    props ?? {},
} as const satisfies ShaderModule<
  SegmentUniforms & SegmentBindings,
  SegmentUniforms,
  SegmentBindings
>;

/** Turn a colour from 0 to 255 into the GPU's 0 to 1, fully opaque. */
function toVec4([red, green, blue]: [number, number, number]): Vec4 {
  return [red / 255, green / 255, blue / 255, 1];
}

/** Colours a BitmapLayer of segment IDs by the scores of the latest search, on the GPU. */
export class SegmentExtension extends LayerExtension {
  static extensionName = "SegmentExtension";
  static defaultProps = {
    scoreTexture: null,
    threshold: SCORE_BYTE_MAX + 1,
    queryId: -1,
  };

  /** Add the segment module to the layer's shaders. */
  getShaders() {
    return { modules: [segmentModule] };
  }

  /** Hand the textures, the threshold and the clicked segment to the GPU before each draw. */
  draw(this: Layer<SegmentExtensionProps & { image: Texture }>) {
    const { image, scoreTexture, threshold, queryId } = this.props;
    if (!scoreTexture) {
      return;
    }
    this.setShaderModuleProps({
      segment: {
        segment_idTexture: image,
        segment_scoreTexture: scoreTexture,
        threshold,
        queryId,
        areaColor: toVec4(AREA_COLOR),
        highlightColor: toVec4(HIGHLIGHT_COLOR),
        queryColor: toVec4(QUERY_COLOR),
      },
    });
  }
}

/** Put one score byte per segment into a texture that the GPU can look up by segment ID. */
export function createScoreTexture(
  device: Device,
  scores: Uint8Array,
): Texture {
  // Rows of SCORE_TEXTURE_WIDTH segments, as many as needed, with the last row padded
  const height = Math.ceil(scores.length / SCORE_TEXTURE_WIDTH);
  const data = new Uint8Array(SCORE_TEXTURE_WIDTH * height);
  data.set(scores);

  // One byte per texel (r8unorm), no mipmaps, so each texel stays one segment's score
  const texture = device.createTexture({
    format: "r8unorm",
    width: SCORE_TEXTURE_WIDTH,
    height,
    mipLevels: 1,
    sampler: { minFilter: "nearest", magFilter: "nearest" },
  });
  texture.writeData(data);
  return texture;
}

/** Decode the ID raster on the CPU too, to find the segment under a click. */
export function decodeSegmentIds(image: ImageBitmap): Uint32Array {
  // Draw the image onto an invisible canvas to read its bytes (R, G, B, A per pixel)
  const canvas = new OffscreenCanvas(image.width, image.height);
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("No 2D canvas to decode the segment IDs");
  }
  context.drawImage(image, 0, 0);
  const rgba = context.getImageData(0, 0, image.width, image.height).data;

  // The same decoding as on the GPU, for every pixel in row order
  const ids = new Uint32Array(image.width * image.height);
  for (let i = 0; i < ids.length; i++) {
    ids[i] = (rgba[4 * i] << 16) | (rgba[4 * i + 1] << 8) | rgba[4 * i + 2];
  }
  return ids;
}
