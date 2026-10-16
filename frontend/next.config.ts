import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  experimental: {
    typedRoutes: true
  },
  async rewrites() {
    const backend = (process.env.INTERNAL_API_BASE_URL || "http://backend:8000")
      .replace(/\/+$/, "")
      .replace(/\/api\/v1$/, "");
    return [
      {
        source: "/api/v1/:path*",
        destination: `${backend}/api/v1/:path*`
      }
    ];
  }
};

export default nextConfig;
