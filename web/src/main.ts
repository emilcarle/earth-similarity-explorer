import "./style.css";
import type { PickingInfo } from "@deck.gl/core";
import type { BitmapLayerPickingInfo } from "@deck.gl/layers";
import type { Device, Texture } from "@luma.gl/core";
import { fetchAreaMetadata, fetchScores, fetchSegmentIds } from "./api";
import { initThresholdSlider, showStatus } from "./controls";
import { createDeck, sceneBounds, segmentLayer } from "./scene";
import { createScoreTexture, decodeSegmentIds } from "./segments";
import { AREA_ID, NO_SEGMENT_ID } from "./settings";

/** Load the area, draw its segments and search on every click. */
async function main(): Promise<void> {
  // The metadata and the ID raster don't depend on each other, so load both at once
  const [metadata, segmentIds] = await Promise.all([
    fetchAreaMetadata(AREA_ID),
    fetchSegmentIds(AREA_ID),
  ]);
  const ids = decodeSegmentIds(segmentIds);
  const bounds = sceneBounds(metadata);

  // What the page shows. Clicks and the slider change it, then render() redraws.
  let device: Device | null = null;
  let scores: Uint8Array = new Uint8Array(metadata.segment_count);
  let scoreTexture: Texture | null = null;
  let queryId = -1;
  let threshold = initThresholdSlider((byte) => {
    threshold = byte;
    render();
  });

  /** Hand the current state to deck.gl and describe it in the status line. */
  function render(): void {
    // deck.gl compares the new layer's props with the old ones and updates only what changed
    deck.setProps({
      layers: [
        segmentLayer(bounds, segmentIds, { scoreTexture, threshold, queryId }),
      ],
    });
    if (queryId < 0) {
      showStatus(`Click a place to find similar ones`);
    } else {
      const highlighted = scores.filter((score) => score >= threshold).length;
      showStatus(
        `Segment ${queryId} · ${highlighted} of ${scores.length} segments highlighted`,
      );
    }
  }

  /** Search for the segment under a click and colour the map by its scores. */
  async function search(info: PickingInfo): Promise<void> {
    // BitmapLayer reports the clicked image pixel. Clicks outside the area hit nothing.
    const pixel = (info as BitmapLayerPickingInfo).bitmap?.pixel;
    if (!pixel || !device) {
      return;
    }
    const segment = ids[pixel[1] * metadata.width + pixel[0]];
    if (segment === NO_SEGMENT_ID) {
      return;
    }

    showStatus(`Searching segment ${segment}…`);
    scores = await fetchScores(AREA_ID, segment);
    // Replace the score texture and free the GPU memory of the old one
    scoreTexture?.destroy();
    scoreTexture = createScoreTexture(device, scores);
    queryId = segment;
    render();
  }

  const deck = createDeck(
    bounds,
    // The GPU is ready, so the texture for "no search yet" can be made
    (readyDevice) => {
      device = readyDevice;
      scoreTexture = createScoreTexture(readyDevice, scores);
      render();
    },
    (info) => {
      search(info).catch((error: unknown) =>
        showStatus(`Search failed. ${error}`),
      );
    },
  );
}

// Show a failed load on the page instead of only in the browser console
main().catch((error: unknown) =>
  showStatus(`Could not load the area. ${error}`),
);
