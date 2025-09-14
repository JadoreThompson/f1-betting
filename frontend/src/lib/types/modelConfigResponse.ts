export interface ModelStats {
  accuracy: number;
  precision: number;
}

export interface ModelConfigResponse {
  params: Record<string, any>;
  features: string[];
  stats: ModelStats;
}
