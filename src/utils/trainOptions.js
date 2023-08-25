import { LuBrainCircuit, LuCircuitBoard } from "react-icons/lu";

export const trainOptions = [
  {
    key: "ml",
    name: "Maching Learning",
    value: "ml",
    icon: LuCircuitBoard,
  },
  {
    key: "ml",
    name: "Deep Learning",
    value: "dl",
    icon: LuBrainCircuit,
  },
];

export const mlOptions = [
  {
    key: "pcasvm",
    label: "PCA + SVM",
    value: "pcasvm",
  },
  {
    key: "ldaknn",
    label: "LDA + KNN",
    value: "ldaknn",
  },
];

export const dlOptions = [
  {
    key: "vgg19",
    label: "VGG19",
    value: "vgg19",
  },
  {
    key: "resnet50",
    label: "ResNet50",
    value: "resnet50",
  },
  {
    key: "convnextbase",
    label: "ConvNeXtBase",
    value: "convnextbase",
  },
];
