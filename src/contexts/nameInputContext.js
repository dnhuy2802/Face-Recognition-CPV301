import { createContext } from "react";

export const initialState = {
  value: "",
  error: false,
  errorMessage: "",
};

export const InputContext = createContext();
