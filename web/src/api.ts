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

/** Ask the search API for a path, and fail with its status if it answers with an error. */
async function get(path: string): Promise<Response> {
  const response = await fetch(`${API_URL}${path}`);
  // fetch only fails when the API can't be reached, so error statuses are checked here
  if (!response.ok) {
    throw new Error(`${path} answered ${response.status}`);
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

/** Score every segment against one, as one byte per segment in segment_id order. */
export async function fetchScores(
  area: string,
  segment: number,
): Promise<Uint8Array> {
  const response = await get(`/areas/${area}/similar?segment=${segment}`);
  return new Uint8Array(await response.arrayBuffer());
}
