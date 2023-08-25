import { getMethod, postMethod } from "../utils/apiHelper";
import { apiUrls } from "../utils/apiUrls";

export async function getModels() {
  const response = await getMethod(apiUrls.predict.getModels);
  if (response.success) {
    return response.data;
  } else {
    throw new Error(response.message);
  }
}

export async function setModel(name) {
  console.log(name);
  const _data = {
    name: name,
  };
  const response = await postMethod(apiUrls.predict.setModel, _data);
  if (response.success) {
    console.log(response.data);
    return response.data;
  } else {
    throw new Error(response.message);
  }
}
