import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";

import { Layout } from "@/components/Layout";
import { AboutPage } from "@/pages/AboutPage";
import { BatchPage } from "@/pages/BatchPage";
import { DiagnosePage } from "@/pages/DiagnosePage";
import { HistoryPage } from "@/pages/HistoryPage";
import { ModelPage } from "@/pages/ModelPage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 2, refetchOnWindowFocus: false, staleTime: 15_000 },
    mutations: { retry: 0 },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<DiagnosePage />} />
            <Route path="batch" element={<BatchPage />} />
            <Route path="history" element={<HistoryPage />} />
            <Route path="model" element={<ModelPage />} />
            <Route path="about" element={<AboutPage />} />
            <Route path="*" element={<DiagnosePage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
