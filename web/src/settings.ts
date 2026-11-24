// Fixed values of the web page, in one place like settings.py in the Python package

// The dev server forwards this path to the search API (see vite.config.ts)
export const API_URL = "/api";
// The only area so far
export const AREA_ID = "schlehdorf";
// Segment IDs are packed into 24 bits (R, G, B), and the largest value marks "no segment"
export const NO_SEGMENT_ID = 2 ** 24 - 1;

// The API sends each score as one byte, with −1 as 0 and 1 as this value
export const SCORE_BYTE_MAX = 255;
// The score texture holds one byte per segment in rows of this many segments
export const SCORE_TEXTURE_WIDTH = 4096;
// The slider starts at byte 230, which stands for a similarity of 0.80
export const DEFAULT_THRESHOLD_BYTE = 230;

// Colours (red, green, blue from 0 to 255) of segments that are not highlighted,
// highlighted segments and the clicked segment
export const AREA_COLOR: [number, number, number] = [62, 68, 74];
export const HIGHLIGHT_COLOR: [number, number, number] = [224, 103, 157];
export const QUERY_COLOR: [number, number, number] = [255, 255, 255];
