import { DEFAULT_THRESHOLD_BYTE, SCORE_BYTE_MAX } from "./settings";

/** Return the element with this id, which index.html must contain. */
function getElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`index.html has no element #${id}`);
  }
  return element as T;
}

/** Show a line of text in the status line. */
export function showStatus(text: string): void {
  getElement("status").textContent = text;
}

/** Set up the map/cloud button and report each press. */
export function initCloudToggle(onPress: () => void): void {
  getElement("cloud-toggle").addEventListener("click", onPress);
}

/** Label the button with where it leads, and disable it while a cloud loads. */
export function setCloudToggle(inCloud: boolean, loading: boolean): void {
  const button = getElement<HTMLButtonElement>("cloud-toggle");
  button.textContent = inCloud ? "Back to map" : "Show cloud";
  button.disabled = loading;
}

/** Describe a threshold byte as the similarity it stands for. */
function describeThreshold(byte: number): string {
  // A byte above 255 is beyond every score, so nothing is highlighted
  if (byte > SCORE_BYTE_MAX) {
    return "nothing";
  }
  // Bytes 0 to 255 stand for similarities from −1 to 1
  return `≥ ${((byte / SCORE_BYTE_MAX) * 2 - 1).toFixed(2)}`;
}

/** Set up the threshold slider in score bytes, report each move and return the start value. */
export function initThresholdSlider(
  onChange: (thresholdByte: number) => void,
): number {
  const slider = getElement<HTMLInputElement>("threshold");
  const label = getElement("threshold-label");

  // One step per score byte, plus one past the top that highlights nothing
  slider.min = "0";
  slider.max = String(SCORE_BYTE_MAX + 1);
  slider.value = String(DEFAULT_THRESHOLD_BYTE);
  label.textContent = describeThreshold(DEFAULT_THRESHOLD_BYTE);

  // "input" fires continuously while the slider moves
  slider.addEventListener("input", () => {
    const byte = Number(slider.value);
    label.textContent = describeThreshold(byte);
    onChange(byte);
  });
  return DEFAULT_THRESHOLD_BYTE;
}
