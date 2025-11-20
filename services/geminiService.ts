import { GoogleGenAI } from "@google/genai";
import { DatasetImage, BBox, LabelClass } from "../types";
import { svgToYolo } from "./yoloService";

const getGeminiClient = () => {
  const apiKey = process.env.API_KEY;
  if (!apiKey) {
    throw new Error("Gemini API Key is missing.");
  }
  return new GoogleGenAI({ apiKey });
};

/**
 * Sends an image to Gemini to detect objects.
 * Note: Gemini 2.5 Flash doesn't natively output exact YOLO coordinates perfectly every time without 
 * fine-tuning, but we can ask it to estimate 2D bounding boxes in 0-1000 scale or normalized.
 * Here we ask for normalized [ymin, xmin, ymax, xmax] which is common for object detection APIs.
 */
export const autoLabelImage = async (
  image: DatasetImage,
  activeLabels: LabelClass[]
): Promise<BBox[]> => {
  try {
    const ai = getGeminiClient();
    
    // Convert blob/url to base64
    const base64Data = await fetch(image.url)
      .then((r) => r.blob())
      .then((blob) => new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64String = (reader.result as string).split(',')[1];
          resolve(base64String);
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      }));

    const labelsList = activeLabels.map(l => l.name).join(', ');

    const prompt = `
      Analyze this image and detect the following objects: ${labelsList}.
      Return a JSON array of detected objects. 
      For each object, provide:
      - "label": the name of the object (must be one of the requested labels)
      - "box_2d": [ymin, xmin, ymax, xmax] where coordinates are normalized between 0 and 1.
      
      Only return the JSON array, no markdown formatting.
    `;

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: {
        parts: [
          { inlineData: { mimeType: 'image/jpeg', data: base64Data } },
          { text: prompt }
        ]
      },
      config: {
        responseMimeType: "application/json"
      }
    });

    const text = response.text;
    if (!text) return [];

    const results = JSON.parse(text);

    const newAnnotations: BBox[] = [];

    if (Array.isArray(results)) {
      results.forEach((res: any) => {
        const labelObj = activeLabels.find(l => l.name.toLowerCase() === res.label.toLowerCase());
        if (labelObj && res.box_2d && res.box_2d.length === 4) {
          const [ymin, xmin, ymax, xmax] = res.box_2d;
          
          // Convert [ymin, xmin, ymax, xmax] to YOLO [cx, cy, w, h]
          // cx = xmin + w/2
          // cy = ymin + h/2
          // w = xmax - xmin
          // h = ymax - ymin
          
          const w = xmax - xmin;
          const h = ymax - ymin;
          const cx = xmin + w / 2;
          const cy = ymin + h / 2;

          newAnnotations.push({
            id: crypto.randomUUID(),
            labelId: labelObj.id,
            x: cx,
            y: cy,
            w: w,
            h: h
          });
        }
      });
    }

    return newAnnotations;

  } catch (error) {
    console.error("Gemini Auto-label error:", error);
    throw error;
  }
};
