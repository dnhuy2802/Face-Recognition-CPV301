import { AiOutlineCloudUpload, AiOutlineSearch } from "react-icons/ai";
import { BiBrain } from "react-icons/bi";
import { appStrings } from "./appStrings";
import { convertLowerCase } from "./utilities";
/// Screens
import UploadScreen from "../pages/Home/Screens/UploadScreen";
import TrainScreen from "../pages/Home/Screens/TrainScreen";
import RecognizeScreen from "../pages/Home/Screens/RecognizeScreen";

function _getItem(key, label, icon, screen) {
  return {
    key,
    label,
    icon,
    screen,
  };
}

export const sidebarOptions = [
  _getItem(
    convertLowerCase(appStrings.navigation.upload),
    appStrings.navigation.upload,
    <AiOutlineCloudUpload />,
    <UploadScreen />
  ),
  _getItem(
    convertLowerCase(appStrings.navigation.train),
    appStrings.navigation.train,
    <BiBrain />,
    <TrainScreen />
  ),
  _getItem(
    convertLowerCase(appStrings.navigation.recognize),
    appStrings.navigation.recognize,
    <AiOutlineSearch />,
    <RecognizeScreen />
  ),
];
