import { postMethod, getMethod, deleteMethod } from "../utils/apiHelper";
import { apiUrls } from "../utils/apiUrls";

export async function uploadImages(name, images) {
  const _data = {
    name: name,
    images: images,
  };
  return await postMethod(apiUrls.upload.images, _data);
}

export async function getFaces() {
  return getMethod(apiUrls.upload.identifiers).then((response) => {
    if (response.success) {
      return response.data;
    } else {
      throw new Error(response.message);
    }
  });
}

export async function deleteFace(name) {
  const _url = apiUrls.upload.delete + "/" + name;
  return await deleteMethod(_url);
}
