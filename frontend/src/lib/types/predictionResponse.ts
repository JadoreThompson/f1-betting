export interface Driver {
  driver_name: string;
  constructor: string;
  nationality: string;
}

export interface PredictionsResponse {
  driver: Driver;
  predicted_position: string;
  predicted_probability: number;
}
