import { useRoutes } from "react-router-dom";
import { routes } from "./routes/routes";

function App() {
  // The useRoutes hook renders defined routes
  return useRoutes(routes);
}

export default App;
