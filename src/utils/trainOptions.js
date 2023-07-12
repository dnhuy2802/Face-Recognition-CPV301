import { LuBrainCircuit, LuCircuitBoard } from "react-icons/lu";

export const trainOptions = [
  {
    key: "1",
    name: "Maching Learning",
    value: "ml",
    icon: LuCircuitBoard,
  },
  {
    key: "2",
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
];
