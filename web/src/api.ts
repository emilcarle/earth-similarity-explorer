import { API_URL } from "./settings";

/** Where an area's ID raster lies in UTM metres, and how many segments it has. */
export type AreaMetadata = {
  west: number;
  north: number;
  pixel_size_m: number;
  width: number;
  height: number;
  segment_count: number;
};

/** A rectangle in the area's UTM coordinates (metres). */
export type BoundingBox = {
  west: number;
  south: number;
  east: number;
  north: number;
};

/** The 3D cloud of the segments in a bbox. */
export type Cloud = {
  // The selected segment IDs in ascending order
  segmentIds: Uint32Array;
  // Three bytes (x, y, z) per selected segment, in the same order, centred at 128
  positions: Uint8Array;
  // Share of the variation along each of the three axes
  explainedVariance: number[];
};

/** Ask the search API for a path, and fail with its message if it answers with an error. */
async function get(path: string): Promise<Response> {
  const response = await fetch(`${API_URL}${path}`);
  // fetch only fails when the API can't be reached, so error statuses are checked here.
  // FastAPI explains its own errors in the detail field of a JSON body.
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : `${path} answered ${response.status}`,
    );
  }
  return response;
}

/** Load where the area's ID raster lies and how many segments it has. */
export async function fetchAreaMetadata(area: string): Promise<AreaMetadata> {
  const response = await get(`/areas/${area}`);
  return response.json();
}

/** Load the area's ID raster as an image whose bytes reach the GPU unchanged. */
export async function fetchSegmentIds(area: string): Promise<ImageBitmap> {
  const response = await get(`/areas/${area}/segment_ids.png`);
  // The pixels are IDs and not colours, so the browser must not adjust them
  return createImageBitmap(await response.blob(), {
    premultiplyAlpha: "none",
    colorSpaceConversion: "none",
  });
}

/** Load each segment's pixel count in segment_id order. */
export async function fetchPixelCounts(area: string): Promise<Uint32Array> {
  const response = await get(`/areas/${area}/pixel_counts`);
  return new Uint32Array(await response.arrayBuffer());
}

/** Score every segment against one, as one byte per segment in segment_id order. */
export async function fetchScores(
  area: string,
  segment: number,
): Promise<Uint8Array> {
  const response = await get(`/areas/${area}/similar?segment=${segment}`);
  return new Uint8Array(await response.arrayBuffer());
}

/** Load the 3D cloud of the segments whose centre lies in the bbox. */
export async function fetchCloud(
  area: string,
  bbox: BoundingBox,
): Promise<Cloud> {
  // The bbox goes into the query string as ?west=…&south=…&east=…&north=…
  const query = new URLSearchParams(
    Object.entries(bbox).map(([key, value]) => [key, String(value)]),
  );
  const response = await get(`/areas/${area}/cloud?${query}`);
  // The body holds 4 bytes per ID, then 3 bytes per position
  const bytes = await response.arrayBuffer();
  const count = bytes.byteLength / (Uint32Array.BYTES_PER_ELEMENT + 3);
  return {
    segmentIds: new Uint32Array(bytes, 0, count),
    positions: new Uint8Array(bytes, Uint32Array.BYTES_PER_ELEMENT * count),
    explainedVariance: (response.headers.get("X-Explained-Variance") ?? "")
      .split(",")
      .map(Number),
  };
}
