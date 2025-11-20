export interface BBox {
  id: string;
  labelId: string;
  x: number; // Normalized center x (0-1)
  y: number; // Normalized center y (0-1)
  w: number; // Normalized width (0-1)
  h: number; // Normalized height (0-1)
}

export interface DatasetImage {
  id: string;
  file: File;
  url: string;
  name: string;
  width: number;
  height: number;
  annotations: BBox[];
  status: 'unlabeled' | 'in-progress' | 'done';
}

export interface LabelClass {
  id: string;
  name: string;
  color: string;
}

export enum ToolMode {
  SELECT = 'SELECT',
  DRAW = 'DRAW',
  PAN = 'PAN'
}

export interface YOLOConfig {
  epochs: number;
  batchSize: number;
  imgSize: number;
  lr0: number;
}
