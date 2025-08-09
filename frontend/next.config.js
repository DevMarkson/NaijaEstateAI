/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  compiler: { removeConsole: process.env.NODE_ENV === 'production' },
  env: {
    NEXT_PUBLIC_API_BASE: process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'
  },
  async rewrites() {
    const apiBase = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';
    return [
      {
        source: '/api/:path*',
        destination: `${apiBase}/:path*`
      }
    ];
  }
};

module.exports = nextConfig;
