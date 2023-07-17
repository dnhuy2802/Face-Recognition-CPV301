import style from "./Camera.module.css";
import { useEffect, useState, useContext } from "react";
import { Select } from "antd";
import Webcam from "react-webcam";
import { StateContext } from "../contexts/stateContext";
import { Typography } from "antd";
import { appStrings } from "../utils/appStrings";
import { VIDEO_WIDTH, VIDEO_HEIGHT } from "../utils/constants";

function Camera({
  isMirror = true,
  showSelectDevice = true,
  mediaCallback = () => {},
}) {
  const [devices, setDevices] = useState([]);
  const { store } = useContext(StateContext);
  const _currentSelectedDevice = store((state) => state.currentCameraId);
  const _setCurrentSelectedDevice = store((state) => state.setCurrentCameraId);

  const videoConstraints = {
    width: VIDEO_WIDTH,
    height: VIDEO_HEIGHT,
    facingMode: "user",
    deviceId: _currentSelectedDevice,
  };

  function selectDevice(deviceId) {
    _setCurrentSelectedDevice(deviceId);
  }

  useEffect(() => {
    /// Get a list of available video input devices.
    navigator.mediaDevices.enumerateDevices().then((devices) => {
      const _videoDevices = devices.filter(
        (device) => device.kind === "videoinput"
      );
      setDevices(_videoDevices);
      /// If selected is null, set the first device as selected
      if (_currentSelectedDevice == null && _videoDevices.length > 0) {
        _setCurrentSelectedDevice(_videoDevices[0].deviceId);
      }
    });
  }, []);

  return (
    <div className={style.container}>
      {_currentSelectedDevice != null ? (
        <Webcam
          className={style.camera}
          audio={false}
          mirrored={isMirror}
          videoConstraints={videoConstraints}
          screenshotFormat="image/jpeg"
          width={VIDEO_WIDTH}
          height={VIDEO_HEIGHT}
          forceScreenshotSourceSize={true}
        >
          {mediaCallback}
        </Webcam>
      ) : (
        <Typography.Text className={style.initCameraText}>
          {appStrings.camera.initCamera}
        </Typography.Text>
      )}
      {showSelectDevice && _currentSelectedDevice ? (
        <Select
          defaultValue={_currentSelectedDevice}
          options={devices.map((device) => ({
            value: device.deviceId,
            label: device.label,
          }))}
          onChange={selectDevice}
          className={style.selectDevice}
        />
      ) : null}
    </div>
  );
}

export default Camera;
