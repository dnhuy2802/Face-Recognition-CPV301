import { createContext } from "react";
import { mlOptions, dlOptions } from "../utils/trainOptions";

/// These variables relate to Backend, be careful when changing it
export const initialState = {
  /// Generals
  faces: [],
  /// ML Configs
  mlTrain: 70,
  mlTest: 30,
  mlAlgorithm: mlOptions[0].value,
  /// DL Configs
  dlTrain: 60,
  dlValid: 20,
  dlTest: 20,
  dlFineTune: 0,
  dlBatchSize: 16,
  dlNetwork: dlOptions[0].value,
};

export const TrainContext = createContext();
