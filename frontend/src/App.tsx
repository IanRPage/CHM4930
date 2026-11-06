import { Route, Routes } from "react-router-dom";
import Layout from "./components/layout/Layout.tsx";
import AboutPage from "./pages/about/AboutPage.tsx";
import BatchPredictionPage from "./pages/batch-prediction/BatchPredictionPage.tsx";
import DatasetExplorerPage from "./pages/dataset-explorer/DatasetExplorerPage.tsx";
import HomePage from "./pages/home/HomePage.tsx";
import NotFoundPage from "./pages/not-found/NotFoundPage.tsx";
import SinglePredictionPage from "./pages/single-prediction/SinglePredictionPage.tsx";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<HomePage />} />
        <Route path="predict" element={<SinglePredictionPage />} />
        <Route path="batch" element={<BatchPredictionPage />} />
        <Route path="explorer" element={<DatasetExplorerPage />} />
        <Route path="about" element={<AboutPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}
