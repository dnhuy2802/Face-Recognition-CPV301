import { postMethod, getMethod, deleteMethod } from "../utils/apiHelper";
import { apiUrls } from "../utils/apiUrls";

/// Upload Faces
export async function uploadFaces(name, images) {
  const _data = {
    name: name,
    images: images,
  };
  return await postMethod(apiUrls.register.new, _data).then((response) => {
    if (response.success) {
      return response.data;
    } else {
      throw new Error(response.message);
    }
  });
}

/// Get All Faces
export async function getFaces() {
  return getMethod(apiUrls.register.faces).then((response) => {
    if (response.success) {
      return response.data;
    } else {
      throw new Error(response.message);
    }
  });
}

/// Delete Face
export async function deleteFace(name) {
  const _url = apiUrls.register.delete + "/" + name;
  return await deleteMethod(_url).then((response) => {
    if (response.success) {
      return response.data;
    } else {
      throw new Error(response.message);
    }
  });
}
