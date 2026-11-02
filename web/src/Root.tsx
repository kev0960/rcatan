import { useEffect, useState } from "react";
import App from "./App";
import TrainingPage from "./training/TrainingPage";

// Hash-based routing: `/training` is already an API path proxied to the server,
// so pages live under `#/...` instead.
type Page = "play" | "training";

function pageFromHash(): Page {
  return window.location.hash.startsWith("#/training") ? "training" : "play";
}

export default function Root() {
  const [page, setPage] = useState<Page>(pageFromHash);

  useEffect(() => {
    const onHash = () => setPage(pageFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  return (
    <>
      <nav className="top-nav">
        <a href="#/" className={page === "play" ? "active" : ""}>
          Play
        </a>
        <a href="#/training" className={page === "training" ? "active" : ""}>
          Training
        </a>
      </nav>
      {page === "play" ? <App /> : <TrainingPage />}
    </>
  );
}
