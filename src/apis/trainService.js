import { postMethod } from "../utils/apiHelper";
import { apiUrls } from "../utils/apiUrls";

export async function startTrain(type) {
  const _data = {
    type: type,
  };
  return postMethod(apiUrls.train.start, _data).then((response) => {
    if (response.success) {
      return response.data;
    } else {
      throw new Error(response.message);
    }
  });
}
