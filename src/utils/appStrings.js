/// This file contains all the strings used in the application

export const appStrings = {
  // App Name
  appName: "Face Recognition",
  // Navigation
  navigation: {
    upload: "Upload",
    train: "Train",
    recognize: "Recognize",
  },
  // Upload
  upload: {
    captureButton: "Register New Face",
    registeredTitle: "Registered Faces",
    searchPlaceholder: "Search",
    noDataFound: "No data found",
    faceCardAddMoreButton: "Add More Images",
    faceCardDeleteButton: "Delete",
    faceCardDeleteConfirmTitle: "Delete Face",
    faceCardDeleteConfirmContent: "Are you sure you want to delete this face?",
    faceCardDeleteConfirmOkButton: "Yes",
    faceCardDeleteConfirmCancelButton: "No",
    modalStartCaptureButton: "Start Capture",
    modalNameInputTitle: "Enter your registing name",
    modalNameInputPlaceholder: "Your name",
    modalNameInputEmptyError: "Name must not be empty",
    modalNameInputDuplicateError: "Name already exists",
    captureButton: "Capture",
    uploadingTitle: "Uploading Images",
  },
  // Camera
  camera: {
    initCamera: "Initializing Camera...",
  },
  // Train
  train: {
    button: "Start Training",
    identityTitle: "Face Identity",
    identityPlaceholder: "Select faces identity",
    ml: {
      title: "Machine Learning",
      algorithmTitle: "Algorithm",
      algorithmPlaceholder: "Select algorithm",
    },
    dl: {
      title: "Deep Learning",
      networkTitle: "Network",
      networkPlaceholder: "Select network",
      fineTuneTitle: "Fine Tune Layers",
      batchSizeTitle: "Batch Size",
    },
    keywords: {
      train: "Train",
      valid: "Valid",
      test: "Test",
    },
  },
};
