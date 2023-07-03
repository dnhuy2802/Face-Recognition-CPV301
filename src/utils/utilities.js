export function debugLogger(message, type) {
  if (process.env.NODE_ENV === "development") {
    switch (type) {
      case "info":
        console.info("--- INFO ---");
        console.info(message);
        break;
      case "error":
        console.error("--- ERROR ---");
        console.error(message);
        break;
      case "warn":
        console.warn("--- WARNING ---");
        console.warn(message);
        break;
      default:
        console.log("--- LOG ---");
        console.log(message);
        break;
    }
  }
}
