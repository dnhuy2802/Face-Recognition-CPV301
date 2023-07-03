import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";
import { BrowserRouter } from "react-router-dom";
import { ConfigProvider } from "antd";
import { appTheme } from "./utils/theme";
import { StateContext, initialState } from "./contexts/stateContext";
import useProviderState from "./hooks/useProviderState";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <ConfigProvider theme={appTheme}>
        <StateContext.Provider
          value={{ store: useProviderState(initialState) }}
        >
          <App />
        </StateContext.Provider>
      </ConfigProvider>
    </BrowserRouter>
  </React.StrictMode>
);
