import "./style.css";
import type { OrbitViewState, PickingInfo } from "@deck.gl/core";
import type { BitmapLayerPickingInfo } from "@deck.gl/layers";
import { PointCloudLayer } from "@deck.gl/layers";
import type { Device, Texture } from "@luma.gl/core";
import {
  fetchAreaMetadata,
  fetchCloud,
  fetchPixelCounts,
  fetchScores,
  fetchSegmentIds,
} from "./api";
import { createPositionTexture } from "./cloud";
import {
  initCloudToggle,
  initThresholdSlider,
  setCloudToggle,
  showStatus,
} from "./controls";
import { buildPoints } from "./points";
import {
  areaViewState,
  createDeck,
  pointLayer,
  sceneBounds,
  segmentLayer,
  toUtmBox,
  visibleBounds,
} from "./scene";
import { createScoreTexture, decodeSegmentIds } from "./segments";
import {
  AREA_ID,
  CLOUD_ROTATION_X,
  MAP_ROTATION_X,
  NO_SEGMENT_ID,
} from "./settings";

/** Load the area, draw its segments, search on every click and switch to the cloud. */
async function main(): Promise<void> {
  // The metadata, the ID raster and the pixel counts don't depend on each other,
  // so load them at once
  const [metadata, segmentIds, pixelCounts] = await Promise.all([
    fetchAreaMetadata(AREA_ID),
    fetchSegmentIds(AREA_ID),
    fetchPixelCounts(AREA_ID),
  ]);
  const ids = decodeSegmentIds(segmentIds);
  const bounds = sceneBounds(metadata);
  // One point per pixel, built once. A new cloud only replaces the position texture.
  const points = buildPoints(ids, pixelCounts, metadata);

  // What the page shows. Clicks, the slider and the toggle change it, then render() redraws.
  let device: Device | null = null;
  let scores: Uint8Array = new Uint8Array(metadata.segment_count);
  let scoreTexture: Texture | null = null;
  let positionTexture: Texture | null = null;
  let explainedVariance: number[] = [];
  let inCloud = false;
  let queryId = -1;
  let threshold = initThresholdSlider((byte) => {
    threshold = byte;
    render();
  });
  // The latest map view, to return to from the cloud
  let mapView: OrbitViewState = areaViewState(bounds, MAP_ROTATION_X);

  /** Hand the current state to deck.gl and describe it in the status line. */
  function render(): void {
    // Both layers stay in the list and only one is visible. deck.gl compares the new
    // layers' props with the old ones and updates only what changed.
    deck.setProps({
      layers: [
        segmentLayer(
          bounds,
          segmentIds,
          { scoreTexture, threshold, queryId },
          !inCloud,
        ),
        pointLayer(
          points,
          {
            positionTexture,
            scoreTexture,
            threshold,
            queryId,
            morph: 1,
            halfWidth: bounds[2],
          },
          inCloud,
        ),
      ],
    });

    // In the cloud, say how much of the variation its three axes show
    const parts: string[] = [];
    if (inCloud) {
      const share = explainedVariance.reduce((sum, value) => sum + value, 0);
      parts.push(
        `The cloud's 3 axes show ${Math.round(share * 100)} % of the variation`,
      );
    }
    if (queryId < 0) {
      parts.push("Click a place to find similar ones");
    } else {
      const highlighted = scores.filter((score) => score >= threshold).length;
      parts.push(
        `Segment ${queryId} · ${highlighted} of ${scores.length} segments highlighted`,
      );
    }
    showStatus(parts.join(". "));
  }

  /** Return the segment under a click on the map or on a point, or null if there is none. */
  function clickedSegment(info: PickingInfo): number | null {
    // A point reports its index in the point arrays
    if (info.layer instanceof PointCloudLayer) {
      return info.index >= 0
        ? points.attributes.getSegmentId.value[info.index]
        : null;
    }
    // The map reports the clicked image pixel. Clicks outside the area hit nothing.
    const pixel = (info as BitmapLayerPickingInfo).bitmap?.pixel;
    const segment = pixel ? ids[pixel[1] * metadata.width + pixel[0]] : null;
    return segment === NO_SEGMENT_ID ? null : segment;
  }

  /** Search for the clicked segment and colour the map and the cloud by its scores. */
  async function search(info: PickingInfo): Promise<void> {
    const segment = clickedSegment(info);
    if (segment === null || !device) {
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

  /** Switch from the map to the cloud of the segments in view, or back to the map. */
  async function toggleCloud(): Promise<void> {
    // Back on the map, the camera returns to where it was and dragging pans again
    if (inCloud) {
      inCloud = false;
      deck.setProps({
        initialViewState: mapView,
        controller: { dragMode: "pan" },
      });
      setCloudToggle(false, false);
      render();
      return;
    }
    if (!device) {
      return;
    }

    // Ask for the cloud of the segments whose centre lies in view
    setCloudToggle(false, true);
    showStatus("Fitting the cloud…");
    const cloud = await fetchCloud(
      AREA_ID,
      toUtmBox(visibleBounds(deck), metadata),
    );
    // Replace the position texture and free the GPU memory of the old one
    positionTexture?.destroy();
    positionTexture = createPositionTexture(
      device,
      metadata.segment_count,
      cloud,
    );
    explainedVariance = cloud.explainedVariance;

    // Tilt the camera towards the cloud, where dragging rotates it
    inCloud = true;
    deck.setProps({
      initialViewState: areaViewState(bounds, CLOUD_ROTATION_X),
      controller: { dragMode: "rotate" },
    });
    setCloudToggle(true, false);
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
  // Keep the latest map view. Each pan or zoom reports the new view.
  deck.setProps({
    onViewStateChange: ({ viewState }) => {
      if (!inCloud) {
        mapView = viewState;
      }
    },
  });
  initCloudToggle(() => {
    // A failed cloud (e.g. too few segments in view) leaves the page on the map
    toggleCloud().catch((error: unknown) => {
      setCloudToggle(false, false);
      showStatus(error instanceof Error ? error.message : String(error));
    });
  });
}

// Show a failed load on the page instead of only in the browser console
main().catch((error: unknown) =>
  showStatus(`Could not load the area. ${error}`),
);
