export const domain = "http://localhost:5000";

export const apiUrls = {
  // Auth
  // Upload
  register: {
    new: "/register_faces/new",
    faces: "/register_faces",
    delete: "/register_faces",
  },
  // Train
  train: {
    start: "/train/start",
  },
  // Predict
  predict: {
    getModels: "/predict/models",
    setModel: "/predict/models/set",
  },
};
