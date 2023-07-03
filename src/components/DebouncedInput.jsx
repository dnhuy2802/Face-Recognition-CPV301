import { useState, useEffect } from "react";
import { Input } from "antd";

function DebouncedInput({ delay = 500, onChange, ...rest }) {
  const [inputValue, setInputValue] = useState("");

  function onInputChange(e) {
    setInputValue(e.target.value);
  }

  useEffect(() => {
    const handler = setTimeout(() => {
      onChange(inputValue);
    }, delay);
    return () => {
      clearTimeout(handler);
    };
  }, [inputValue]);

  return <Input onChange={onInputChange} {...rest} />;
}

export default DebouncedInput;
