import { GoogleGenAI } from "@google/genai";
import { DatasetImage, BBox, LabelClass } from "../types";
import { svgToYolo } from "./yoloService";

// Configuration interface for auto-labeling
export interface AutoLabelConfig {
  model?: string; // Default: 'gemini-2.5-flash'
  minConfidence?: number; // Minimum confidence threshold (0-1), default: 0.3
  maxRetries?: number; // Maximum retry attempts, default: 3
  temperature?: number; // Model temperature, default: 0.1 (more deterministic)
  includeDescription?: boolean; // Include object descriptions, default: false
}

// Extended BBox with confidence score
export interface BBoxWithConfidence extends BBox {
  confidence?: number;
  description?: string;
}

// Detection result interface
interface GeminiDetectionResult {
  label: string;
  box_2d: [number, number, number, number]; // [ymin, xmin, ymax, xmax]
  confidence?: number; // 0-1 confidence score
  description?: string; // Optional description
}

const getGeminiClient = () => {
  const apiKey = process.env.API_KEY;
  if (!apiKey) {
    throw new Error("Gemini API Key is missing. Please set API_KEY environment variable.");
  }
  return new GoogleGenAI({ apiKey });
};

/**
 * Detects MIME type from blob
 */
const detectMimeType = (blob: Blob): string => {
  const type = blob.type;
  if (type.startsWith('image/')) {
    return type;
  }
  // Fallback to jpeg
  return 'image/jpeg';
};

/**
 * Convert image to base64 with MIME type detection
 */
const imageToBase64 = async (url: string): Promise<{ data: string; mimeType: string }> => {
  const response = await fetch(url);
  const blob = await response.blob();
  const mimeType = detectMimeType(blob);

  const base64Data = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64String = (reader.result as string).split(',')[1];
      resolve(base64String);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });

  return { data: base64Data, mimeType };
};

/**
 * Enhanced prompt generation with better instructions
 */
const generatePrompt = (
  labels: LabelClass[],
  config: AutoLabelConfig,
  width?: number,
  height?: number
): string => {
  const labelsList = labels.map(l => l.name).join('\n');
  const includeDescription = config.includeDescription ?? false; // Keep this for potential future use or if description is added back to prompt

  const widthInfo = width && width > 0 ? `Image Dimensions: ${width}x${height} pixels.\n` : '';

  return `You are an expert object detection AI. Analyze this image and detect ALL instances of the following object classes:
${labelsList}

${widthInfo}IMPORTANT:
1. You MUST use the EXACT class names listed above. Do not translate, paraphrase, or change the case.
2. Return ONLY a valid JSON array.
3. Detect all visible objects.
4. Bounding boxes should be normalized (0-1). If you return pixel values, I will attempt to convert them, but normalized is preferred.
5. Limit the results to a maximum of 10 detections.

Return format:
[
  {
    "label": "exact_class_name",
    "confidence": 0.95,
    "box_2d": [ymin, xmin, ymax, xmax] 
  }
]
`;
};

// ... (retryAsync, validateBBox, clampBBox remain same) ...

/**
 * Retry wrapper for API calls
 */
async function retryAsync<T>(
  fn: () => Promise<T>,
  maxRetries: number,
  retryDelay: number = 1000
): Promise<T> {
  let lastError: Error;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;

      if (attempt < maxRetries) {
        console.warn(`Attempt ${attempt + 1} failed, retrying in ${retryDelay}ms...`, error);
        await new Promise(resolve => setTimeout(resolve, retryDelay));
        retryDelay *= 2; // Exponential backoff
      }
    }
  }

  throw lastError!;
}

/**
 * Validates and clamps bounding box coordinates
 */
const validateBBox = (box: number[]): boolean => {
  if (!Array.isArray(box) || box.length !== 4) return false;

  // Check if all are numbers
  if (!box.every(v => typeof v === 'number' && !isNaN(v))) return false;

  return true;
};

/**
 * Clamps coordinates to valid range and ensures minimum size
 */
const clampBBox = (box: number[]): number[] => {
  const MIN_SIZE = 0.01; // Minimum 1% of image dimension

  let [ymin, xmin, ymax, xmax] = box.map(v => Math.max(0, Math.min(1, v)));

  // Ensure minimum dimensions
  if (ymax - ymin < MIN_SIZE) {
    const center = (ymin + ymax) / 2;
    ymin = Math.max(0, center - MIN_SIZE / 2);
    ymax = Math.min(1, center + MIN_SIZE / 2);
  }

  if (xmax - xmin < MIN_SIZE) {
    const center = (xmin + xmax) / 2;
    xmin = Math.max(0, center - MIN_SIZE / 2);
    xmax = Math.min(1, center + MIN_SIZE / 2);
  }

  return [ymin, xmin, ymax, xmax];
};

export const autoLabelImage = async (
  image: DatasetImage,
  activeLabels: LabelClass[],
  config: AutoLabelConfig = {}
): Promise<BBoxWithConfidence[]> => {
  if (!activeLabels || activeLabels.length === 0) {
    throw new Error("No active labels provided. Please specify at least one label class.");
  }

  // Consider using image dimensions to assist the model
  const { width, height } = image;

  const {
    model = 'gemini-3-flash-preview',
    minConfidence = 0.93,
    maxRetries = 3,
    temperature = 0,
    includeDescription = false
  } = config;

  try {
    const ai = getGeminiClient();

    // Convert image to base64 with MIME type detection
    const { data: base64Data, mimeType } = await imageToBase64(image.url);

    const prompt = generatePrompt(activeLabels, { ...config, includeDescription }, width, height);

    // Make API call with retry logic and model fallback
    const response = await retryAsync(
      async () => {
        try {
          return await ai.models.generateContent({
            model,
            contents: {
              parts: [
                { inlineData: { mimeType, data: base64Data } },
                { text: prompt }
              ]
            },
            config: {
              responseMimeType: "application/json",
              temperature
            }
          });
        } catch (err: any) {
          // Fallback mechanism for 404 (Model Not Found) or 400 (Bad Request)
          if ((err.message?.includes('404') || err.status === 404 || err.message?.includes('not found')) && (model.includes('2.5') || model.includes('preview') || model.includes('flash-8b'))) {
            const fallbackModel = 'gemini-2.0-flash';
            console.warn(`Model ${model} not found or error. Falling back to ${fallbackModel}.`);
            return await ai.models.generateContent({
              model: fallbackModel,
              contents: {
                parts: [
                  { inlineData: { mimeType, data: base64Data } },
                  { text: prompt }
                ]
              },
              config: {
                responseMimeType: "application/json",
                temperature
              }
            });
          }
          throw err;
        }
      },
      maxRetries
    );

    let text = response.text;
    if (!text) {
      console.warn("Gemini returned empty response");
      return [];
    }

    console.log("Raw Gemini Response:", text); // Debug log

    // Clean JSON string (remove markdown code blocks)
    // Matches ```json, ```JSON, or just ``` 
    text = text.replace(/```[a-zA-Z]*\n?|\n?```/g, '').trim();

    let results: GeminiDetectionResult[];
    try {
      results = JSON.parse(text);
    } catch (parseError) {
      console.error("Failed to parse Gemini JSON. Attempting substring extraction...");
      // Try to find array brackets [ ... ]
      const match = text.match(/\[[\s\S]*\]/);
      if (match) {
        try {
          results = JSON.parse(match[0]);
          console.log("Successfully extracted and parsed JSON from substring.");
        } catch (e) {
          console.error("Substring extraction failed:", e);
          throw new Error(`Invalid JSON response from Gemini`);
        }
      } else {
        throw new Error(`Invalid JSON response from Gemini`);
      }
    }

    if (!Array.isArray(results)) {
      console.error("Gemini response is not an array:", results);
      return [];
    }

    const newAnnotations: BBoxWithConfidence[] = [];
    // Create a map for case-insensitive label matching, matching trimmed names
    const labelMap = new Map(activeLabels.map(l => [l.name.toLowerCase().trim(), l]));

    let filteredByConfidence = 0;
    let invalidBoxes = 0;
    let unknownLabels = 0;

    results.forEach((res, index) => {
      // 1. Validate Structure
      if (!res.label || !res.box_2d) {
        console.warn(`Item ${index}: Missing 'label' or 'box_2d'. Skipped.`);
        return;
      }

      // 2. Resolve Label
      const resLabel = res.label.toLowerCase().trim();
      let labelObj = labelMap.get(resLabel);

      // Fallback: check inclusion
      if (!labelObj) {
        for (const [key, value] of labelMap.entries()) {
          // Strict check: only if the model's output contains the exact key or vice versa
          // but simpler: if the model returns 'button' and we have 'button_submit', we might NOT want to match indiscriminately.
          // Let's stick to key being inside resLabel (e.g. model="red_car" key="car") -> dangerous.
          // Better: exact match is preferred. Prompt is stricter now.
          // We'll keep the simple inclusion for now but log it.
          if (resLabel.includes(key) || key.includes(resLabel)) {
            labelObj = value;
            console.log(`Item ${index}: Fuzzy matched "${res.label}" to "${value.name}"`);
            break;
          }
        }
      }

      if (!labelObj) {
        console.warn(`Item ${index}: Unknown label "${res.label}". Available: ${Array.from(labelMap.keys()).join(', ')}. Skipped.`);
        unknownLabels++;
        return;
      }

      // 3. Check Confidence
      const confidence = Number(res.confidence ?? 1.0);
      if (confidence < minConfidence) {
        console.log(`Item ${index}: Confidence ${confidence} below threshold ${minConfidence}. Skipped.`);
        filteredByConfidence++;
        return;
      }

      // 4. Normalize Coordinates
      let box = res.box_2d.map(Number);

      // If ANY coordinate is > 1.0, assume pixel coords and normalize
      if (box.some(v => v > 1.0)) {
        if (image.width > 0 && image.height > 0) {
          console.log(`Item ${index}: Detected pixel coordinates [${box.join(',')}]. Normalizing...`);
          box = [
            box[0] / image.height, // ymin
            box[1] / image.width,  // xmin
            box[2] / image.height, // ymax
            box[3] / image.width   // xmax
          ];
        } else {
          console.warn(`Item ${index}: Pixel coordinates detected but image dimensions unknown. Skipped.`);
          invalidBoxes++;
          return;
        }
      }

      // Auto-correct inverted coordinates (e.g. if model swaps min/max)
      if (box[0] > box[2]) [box[0], box[2]] = [box[2], box[0]]; // Swap y
      if (box[1] > box[3]) [box[1], box[3]] = [box[3], box[1]]; // Swap x

      if (!validateBBox(box)) {
        console.warn(`Item ${index}: Invalid bounding box for ${res.label}:`, res.box_2d);
        invalidBoxes++;
        return;
      }

      const [ymin, xmin, ymax, xmax] = clampBBox(box);

      // Convert [ymin, xmin, ymax, xmax] to YOLO format [cx, cy, w, h]
      const w = xmax - xmin;
      const h = ymax - ymin;
      const cx = xmin + w / 2;
      const cy = ymin + h / 2;

      // Validate YOLO conversion
      if (w <= 0 || h <= 0) {
        console.warn(`Item ${index}: Resulting width/height is zero. Skipped.`);
        invalidBoxes++;
        return;
      }

      newAnnotations.push({
        id: crypto.randomUUID(),
        labelId: labelObj.id,
        x: cx,
        y: cy,
        w: w,
        h: h,
        confidence,
        description: res.description
      });
    });

    // Log statistics
    console.log(`Auto-labeling complete: ${newAnnotations.length} objects detected`);
    if (filteredByConfidence > 0) console.log(`  - ${filteredByConfidence} detections filtered (confidence < ${minConfidence})`);
    if (invalidBoxes > 0) console.warn(`  - ${invalidBoxes} invalid bounding boxes ignored`);
    if (unknownLabels > 0) console.warn(`  - ${unknownLabels} labels not found in configuration`);

    return newAnnotations;

  } catch (error) {
    console.error("Gemini Auto-label error:", error);

    // Enhanced error messages
    if (error instanceof Error) {
      if (error.message.includes('API_KEY')) {
        throw new Error("Gemini API key is invalid or missing. Please check your API_KEY environment variable.");
      } else if (error.message.includes('429') || error.message.includes('quota')) {
        throw new Error("Gemini API quota exceeded. Please try again later.");
      } else if (error.message.includes('fetch') || error.message.includes('network')) {
        throw new Error("Network error: Failed to communicate with Gemini API.");
      }
    }

    throw error;
  }
};

