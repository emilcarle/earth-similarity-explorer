// Fixed values of the web page, in one place like settings.py in the Python package

// The dev server forwards this path to the search API (see vite.config.ts)
export const API_URL = "/api";
// The only area so far
export const AREA_ID = "schlehdorf";
// Segment IDs are packed into 24 bits (R, G, B), and the largest value marks "no segment"
export const NO_SEGMENT_ID = 2 ** 24 - 1;

// The API sends each score as one byte, with −1 as 0 and 1 as this value
export const SCORE_BYTE_MAX = 255;
// The score and position textures hold one texel per segment in rows of this many segments
export const SEGMENT_TEXTURE_WIDTH = 4096;
// The API sends each cloud position as one byte per axis, with the centre of the cloud
// at 128 and the edge of its range at 128 ± 127
export const POSITION_BYTE_CENTRE = 128;
export const POSITION_BYTE_HALF_RANGE = 127;
// Spread (standard deviation) of each point's random offset in the cloud, as a share of
// the cloud's half-width, so a segment shows as a little cloud of its pixels
export const CLOUD_JITTER = 0.02;
// Camera tilt (rotationX in degrees) on the map, which looks straight down, and in the
// cloud, which is seen from the side at a slant
export const MAP_ROTATION_X = 90;
export const CLOUD_ROTATION_X = 30;
// Size of a point on the screen in pixels
export const POINT_SIZE_PX = 2;
// The point cloud draws one point per pixel up to this many points, a count that
// rotated smoothly on an M4 laptop. Larger areas are thinned evenly.
export const MAX_POINTS = 1_000_000;
// The slider starts at byte 230, which stands for a similarity of 0.80
export const DEFAULT_THRESHOLD_BYTE = 230;

// Colours (red, green, blue from 0 to 255) of segments that are not highlighted,
// highlighted segments and the clicked segment
export const AREA_COLOR: [number, number, number] = [62, 68, 74];
export const HIGHLIGHT_COLOR: [number, number, number] = [224, 103, 157];
export const QUERY_COLOR: [number, number, number] = [255, 255, 255];
