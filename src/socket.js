import { io } from "socket.io-client";
import { domain } from "./utils/apiUrls";

export const socket = io(domain);
