import { createContext } from "react";

/// These variables relate to Backend, be careful when changing it
export const initialState = {
  models: [],
  currentModel: null,
};

export const PredictContext = createContext();
