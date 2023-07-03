/// API Helper
/// This file contains the helper functions for API calls

import axios from "axios";
import { debugLogger } from "./utilities";

/// Base API instance including the base URL and timeout
const apiInstance = axios.create({
  baseURL: "http://localhost:3000",
  timeout: 1000,
});

/// GET method
export function getMethod(url, config) {
  try {
    return apiInstance.get(url, config);
  } catch (error) {
    debugLogger(error, "error");
    throw new Error(error);
  }
}

/// POST method
export function postMethod(url, data, config) {
  try {
    return apiInstance.post(url, data, config);
  } catch (error) {
    debugLogger(error, "error");
    throw new Error(error);
  }
}

/// PUT method
export function putMethod(url, data, config) {
  try {
    return apiInstance.put(url, data, config);
  } catch (error) {
    debugLogger(error, "error");
    throw new Error(error);
  }
}

/// DELETE method
export function deleteMethod(url, config) {
  try {
    return apiInstance.delete(url, config);
  } catch (error) {
    debugLogger(error, "error");
    throw new Error(error);
  }
}
