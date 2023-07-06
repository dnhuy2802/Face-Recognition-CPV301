import HomePage from "../pages/Home/HomePage";
import CapturePage from "../pages/Home/CapturePage";

export const routes = [
  {
    path: "/",
    element: <HomePage />,
  },
  {
    path: "/capture",
    element: <CapturePage />,
  },
];
