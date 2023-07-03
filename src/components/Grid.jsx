const gridStyle = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
  gridGap: "1rem",
};

function Grid({ children }) {
  return <div style={gridStyle}>{children}</div>;
}

export default Grid;
