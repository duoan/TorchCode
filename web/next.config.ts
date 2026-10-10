import type { NextConfig } from "next";

const dashboard = process.env.TORCHCODE_DASHBOARD === "1";

const nextConfig: NextConfig = {
  // Separate dev output avoids a shared Next.js lock between ports 3000/4000.
  distDir: dashboard ? ".next-practice" : ".next",
  async rewrites() {
    return {
      beforeFiles: dashboard
        ? [{ source: "/", destination: "/practice" }]
        : [],
      afterFiles: [],
      fallback: [],
    };
  },
};

export default nextConfig;
