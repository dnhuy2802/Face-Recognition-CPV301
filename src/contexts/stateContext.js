import { createContext } from "react";
import { sidebarOptions } from "../utils/sidebarOptions";

export const initialState = {
  /// General
  user: null,
  isAuth: false,
  isLoading: false,
  /// Sidebar
  selectedSidebarItem: sidebarOptions[0].key,
  /// Register Faces
  currentCameraId: null,
  userUploadImages: [],
  userUploadName: "",
  /// Training
  isTraining: false,
  faceOptions: [],
};

export const StateContext = createContext();
