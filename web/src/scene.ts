import type { OrbitViewState, PickingInfo } from "@deck.gl/core";
import { Deck, OrbitView } from "@deck.gl/core";
import { BitmapLayer, PointCloudLayer } from "@deck.gl/layers";
import type { Device } from "@luma.gl/core";
import type { AreaMetadata, BoundingBox } from "./api";
import { CloudExtension, type CloudExtensionProps } from "./cloud";
import type { Points } from "./points";
import { SegmentExtension, type SegmentExtensionProps } from "./segments";
import { MAP_ROTATION_X, POINT_SIZE_PX } from "./settings";

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

/** Return the camera's view of the whole area, tilted by rotationX (90 looks straight down). */
export function areaViewState(
  bounds: Bounds,
  rotationX: number,
): OrbitViewState {
  // At zoom 0 one metre is one screen pixel, and each zoom step doubles the size.
  // Fit the area into 90 % of the window. The cloud is as wide as the area, so it fits too.
  const [left, bottom, right, top] = bounds;
  const zoom = Math.log2(
    0.9 *
      Math.min(
        window.innerWidth / (right - left),
        window.innerHeight / (top - bottom),
      ),
  );
  return { target: [0, 0, 0], rotationX, rotationOrbit: 0, zoom };
}

/** Create the 3D scene seen from straight above. Its layers are set later with setProps. */
export function createDeck(
  bounds: Bounds,
  onDeviceInitialized: (device: Device) => void,
  onClick: (info: PickingInfo) => void,
): Deck<OrbitView> {
  // One 3D camera that orbits around the Z axis, for both the map and the cloud.
  // On the map it looks straight down, and dragging pans instead of rotating.
  return new Deck({
    views: new OrbitView({ orbitAxis: "Z", fovy: 30 }),
    initialViewState: areaViewState(bounds, MAP_ROTATION_X),
    controller: { dragMode: "pan" },
    onDeviceInitialized,
    onClick,
  });
}

/** Return the part of the map plane that the camera sees from straight above, in scene coordinates. */
export function visibleBounds(deck: Deck<OrbitView>): Bounds {
  // Project the four screen corners back onto the map plane (z 0)
  const viewport = deck.getViewports()[0];
  const corners = [
    [0, 0],
    [viewport.width, 0],
    [0, viewport.height],
    [viewport.width, viewport.height],
  ].map((corner) => viewport.unproject(corner));
  const xs = corners.map(([x]) => x);
  const ys = corners.map(([, y]) => y);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

/** Turn scene bounds into a bbox in UTM metres by undoing the shift of sceneBounds. */
export function toUtmBox(
  [left, bottom, right, top]: Bounds,
  metadata: AreaMetadata,
): BoundingBox {
  // The area's centre in UTM, which is 0, 0 in the scene
  const centreX = metadata.west + (metadata.width * metadata.pixel_size_m) / 2;
  const centreY =
    metadata.north - (metadata.height * metadata.pixel_size_m) / 2;
  return {
    west: left + centreX,
    south: bottom + centreY,
    east: right + centreX,
    north: top + centreY,
  };
}

/** The ID raster stretched over the area and coloured by the segment extension. */
export function segmentLayer(
  bounds: Bounds,
  segmentIds: ImageBitmap,
  props: SegmentExtensionProps,
  visible: boolean,
): BitmapLayer<SegmentExtensionProps> {
  return new BitmapLayer<SegmentExtensionProps>({
    id: "segments",
    visible,
    image: segmentIds,
    bounds,
    // Clicks report which pixel of the image was hit
    pickable: true,
    extensions: [new SegmentExtension()],
    ...props,
  });
}

/** The pixels as points that the cloud extension moves between map and cloud. */
export function pointLayer(
  points: Points,
  props: CloudExtensionProps,
  visible: boolean,
): PointCloudLayer<unknown, CloudExtensionProps> {
  return new PointCloudLayer<unknown, CloudExtensionProps>({
    id: "points",
    // A hidden layer keeps its GPU buffers, so the points are uploaded only once
    visible,
    data: points,
    pointSize: POINT_SIZE_PX,
    sizeUnits: "pixels",
    // No lighting, so a point shows its colour as it is
    material: false,
    // Clicks report the index of the point that was hit
    pickable: true,
    extensions: [new CloudExtension()],
    ...props,
  });
}
