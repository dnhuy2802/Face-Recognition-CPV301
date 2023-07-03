function Flex({
  children,
  direction = "row",
  align = "start",
  justify = "start",
}) {
  const flexStyle = {
    display: "flex",
    flexDirection: direction,
    alignItems: align,
    justifyContent: justify,
  };

  return <div style={flexStyle}>{children}</div>;
}

export default Flex;
