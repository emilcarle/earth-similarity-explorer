import type { PickingInfo } from "@deck.gl/core";
import { Deck, OrbitView } from "@deck.gl/core";
import { BitmapLayer } from "@deck.gl/layers";
import type { Device } from "@luma.gl/core";
import type { AreaMetadata } from "./api";
import { SegmentExtension, type SegmentExtensionProps } from "./segments";

/** [left, bottom, right, top] in scene coordinates */
type Bounds = [number, number, number, number];

/** Return the ID raster's extent in scene coordinates. */
export function sceneBounds(metadata: AreaMetadata): Bounds {
  // Scene coordinates are UTM metres with the area's centre at 0, 0. Small numbers
  // stay precise in the GPU's 32-bit arithmetic, where UTM values would lose metres.
  const halfWidth = (metadata.width * metadata.pixel_size_m) / 2;
  const halfHeight = (metadata.height * metadata.pixel_size_m) / 2;
  return [-halfWidth, -halfHeight, halfWidth, halfHeight];
}

/** Create the 3D scene seen from straight above. Its layers are set later with setProps. */
export function createDeck(
  bounds: Bounds,
  onDeviceInitialized: (device: Device) => void,
  onClick: (info: PickingInfo) => void,
): Deck<OrbitView> {
  // At zoom 0 one metre is one screen pixel, and each zoom step doubles the size.
  // Fit the area into 90 % of the window.
  const [left, bottom, right, top] = bounds;
  const zoom = Math.log2(
    0.9 *
      Math.min(
        window.innerWidth / (right - left),
        window.innerHeight / (top - bottom),
      ),
  );

  // One 3D camera that orbits around the Z axis, so the point cloud can use it later.
  // rotationX 90 looks straight down, and dragging pans instead of rotating.
  return new Deck({
    views: new OrbitView({ orbitAxis: "Z", fovy: 30 }),
    initialViewState: {
      target: [0, 0, 0],
      rotationX: 90,
      rotationOrbit: 0,
      zoom,
    },
    controller: { dragMode: "pan" },
    onDeviceInitialized,
    onClick,
  });
}

/** The ID raster stretched over the area and coloured by the segment extension. */
export function segmentLayer(
  bounds: Bounds,
  segmentIds: ImageBitmap,
  props: SegmentExtensionProps,
): BitmapLayer<SegmentExtensionProps> {
  return new BitmapLayer<SegmentExtensionProps>({
    id: "segments",
    image: segmentIds,
    bounds,
    // Clicks report which pixel of the image was hit
    pickable: true,
    extensions: [new SegmentExtension()],
    ...props,
  });
}
