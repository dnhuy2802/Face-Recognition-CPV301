/// API Helper
/// This file contains the helper functions for API calls

import axios from "axios";
import { debugLogger } from "./utilities";
import { domain } from "./apiUrls";

/// Base API instance including the base URL and timeout
const apiInstance = axios.create({
  baseURL: `${domain}/apis`,
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json",
  },
});

const defaultConfig = {
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json",
  },
};

/// GET method
export async function getMethod(url, config = defaultConfig) {
  try {
    const response = await apiInstance.get(url, config);
    return response.data;
  } catch (error) {
    debugLogger(error, "error");
    throw new Error(error);
  }
}

/// POST method
export async function postMethod(url, data, config = defaultConfig) {
  try {
    const response = await apiInstance.post(url, data, config);
    return response.data;
  } catch (error) {
    debugLogger(error, "error");
    throw new Error(error);
  }
}

/// PUT method
export async function putMethod(url, data, config = defaultConfig) {
  try {
    const response = await apiInstance.put(url, data, config);
    return response.data;
  } catch (error) {
    debugLogger(error, "error");
    throw new Error(error);
  }
}

/// DELETE method
export async function deleteMethod(url, config = defaultConfig) {
  try {
    const response = await apiInstance.delete(url, config);
    return response.data;
  } catch (error) {
    debugLogger(error, "error");
    throw new Error(error);
  }
}
