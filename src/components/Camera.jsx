import style from "./Camera.module.css";
import { useEffect, useState, useContext, useRef } from "react";
import { Select } from "antd";
import Webcam from "react-webcam";
import { StateContext } from "../contexts/stateContext";
import { Typography } from "antd";
import { appStrings } from "../utils/appStrings";

function Camera({
  isMirror = true,
  showSelectDevice = true,
  mediaCallback = () => {},
}) {
  const [devices, setDevices] = useState([]);
  const { store } = useContext(StateContext);
  const _currentSelectedDevice = store((state) => state.devideId);
  const _setCurrentSelectedDevice = store((state) => state.setDevideId);

  const videoConstraints = {
    width: 1280,
    height: 720,
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
      if (!_currentSelectedDevice && _videoDevices.length > 0) {
        _setCurrentSelectedDevice(_videoDevices[0].deviceId);
      }
    });
  }, []);

  return (
    <div className={style.container}>
      {_currentSelectedDevice ? (
        <Webcam
          className={style.camera}
          audio={false}
          mirrored={isMirror}
          videoConstraints={videoConstraints}
          screenshotFormat="image/jpeg"
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
