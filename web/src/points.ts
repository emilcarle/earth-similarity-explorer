import type { AreaMetadata } from "./api";
import { MAX_POINTS, NO_SEGMENT_ID } from "./settings";

/** Per-point arrays in the binary form a PointCloudLayer reads without a loop per point. */
export type Points = {
  length: number;
  attributes: {
    // Map position in scene coordinates (x, y, z)
    getPosition: { value: Float32Array; size: 3 };
    // The segment under the point (float32 holds IDs below 2^24 exactly)
    getSegmentId: { value: Float32Array; size: 1 };
    // Random offset in the cloud (x, y, z), standard normal before scaling
    getJitter: { value: Float32Array; size: 3 };
  };
};

/** Draw a random number from the standard normal distribution (Box–Muller). */
function randomNormal(): number {
  // 1 − random() lies in (0, 1], so the logarithm never sees 0
  return (
    Math.sqrt(-2 * Math.log(1 - Math.random())) *
    Math.cos(2 * Math.PI * Math.random())
  );
}

/** Scatter one random point per pixel over the ID raster's cells, at most MAX_POINTS. */
export function buildPoints(
  ids: Uint32Array,
  pixelCounts: Uint32Array,
  metadata: AreaMetadata,
): Points {
  const { width, pixel_size_m: cellSize } = metadata;
  // The raster's top-left corner in scene coordinates (area centre at 0, 0)
  const left = -(width * cellSize) / 2;
  const top = (metadata.height * cellSize) / 2;

  // Calculate the share of pixels that get a point (1 unless the area exceeds MAX_POINTS)
  let pixelTotal = 0;
  for (const id of ids) {
    pixelTotal += id === NO_SEGMENT_ID ? 0 : pixelCounts[id];
  }
  const share = Math.min(1, MAX_POINTS / pixelTotal);
  // Points per cell, rounded up or down at random so that every cell keeps its share on
  // average (a 3-pixel cell at share 0.1 gets one point 30 % of the time)
  const cellPoints = ids.map((id) =>
    id === NO_SEGMENT_ID
      ? 0
      : Math.floor(pixelCounts[id] * share + Math.random()),
  );
  const length = cellPoints.reduce((sum, count) => sum + count, 0);
  const positions = new Float32Array(length * 3);
  const segmentIds = new Float32Array(length);
  const jitter = new Float32Array(length * 3);

  // Fill the points cell by cell, with z at 0 on the map. The 100 m raster doesn't
  // show which of a cell's pixels lie in the area, so they are scattered over all of it.
  let point = 0;
  ids.forEach((id, cell) => {
    const column = cell % width;
    const row = Math.floor(cell / width);
    for (let k = 0; k < cellPoints[cell]; k++, point++) {
      positions[3 * point] = left + (column + Math.random()) * cellSize;
      positions[3 * point + 1] = top - (row + Math.random()) * cellSize;
      segmentIds[point] = id;
      jitter.set([randomNormal(), randomNormal(), randomNormal()], 3 * point);
    }
  });
  return {
    length,
    attributes: {
      getPosition: { value: positions, size: 3 },
      getSegmentId: { value: segmentIds, size: 1 },
      getJitter: { value: jitter, size: 3 },
    },
  };
}
