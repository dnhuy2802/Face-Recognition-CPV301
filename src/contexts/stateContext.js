import { createContext } from "react";
import { sidebarOptions } from "../utils/sidebarOptions";

export const initialState = {
  /// General
  user: null,
  isAuth: false,
  isLoading: false,
  /// Sidebar
  selectedSidebarItem: sidebarOptions[0].key,
  /// Camera
  devideId: null,
  images: [],
  /// Upload
  uploadName: "",
};

export const StateContext = createContext();
