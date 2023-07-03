function Flex({ children, direction, align, justify }) {
  const flexStyle = {
    display: "flex",
    flexWrap: "wrap",
    flexDirection: direction,
    alignItems: align,
    justifyContent: justify,
  };

  return <div style={flexStyle}>{children}</div>;
}

export default Flex;
