# QuickScore Frontend

React-based frontend for the QuickScore Pre-Seed Startup Analyzer platform.

## Features

- 🔐 **Authentication System** - JWT-based login/register with protected routes
- 📊 **Dashboard** - Startup analysis with real-time progress tracking
- 🚀 **Batch Processing** - Analyze multiple startups simultaneously
- 🔍 **Comparable Startups** - Find similar startups with multi-dimensional matching
- 📈 **Monitoring Dashboard** - System health and performance metrics
- 📱 **Responsive Design** - Mobile-first design with Tailwind CSS
- ⚡ **Real-time Updates** - WebSocket integration for live notifications

## Tech Stack

- **React 18** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **React Router** for navigation
- **React Query** for server state management
- **React Hook Form** for form handling
- **Zustand** for client state management
- **Recharts** for data visualization
- **Framer Motion** for animations
- **Socket.io** for real-time communication

## Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── ui/             # Basic UI elements
│   ├── charts/         # Chart components
│   └── forms/          # Form components
├── pages/              # Page components
├── contexts/           # React contexts
├── hooks/              # Custom hooks
├── lib/                # Utilities and API client
├── types/              # TypeScript type definitions
└── styles/             # Global styles
```

## Development Setup

### Prerequisites

- Node.js 18+ and npm/yarn
- Backend API running on http://localhost:8000

### Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create environment file:
```bash
cp .env.example .env.local
```

4. Configure environment variables:
```env
VITE_API_URL=http://localhost:8000/api
VITE_WS_URL=ws://localhost:8000
VITE_APP_NAME=QuickScore
VITE_APP_VERSION=1.0.0
```

5. Start development server:
```bash
npm run dev
```

The application will be available at http://localhost:3000

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript type checking

## API Integration

The frontend communicates with the backend through a comprehensive API client located in `src/lib/api.ts`. The client handles:

- **Authentication** - Login, register, token management
- **Startup Management** - CRUD operations for startups
- **Analysis** - Create and monitor analysis jobs
- **Batch Processing** - Manage batch analysis jobs
- **Comparable Startups** - Find and compare similar startups
- **Monitoring** - System health and performance metrics
- **File Upload** - Handle pitch deck and CSV uploads

## State Management

### Authentication State
Managed by `AuthContext` with persistent login state and automatic token refresh.

### Server State
Uses React Query for efficient server state management with caching, background updates, and optimistic updates.

### Client State
Uses Zustand for lightweight client-side state management.

## Real-time Features

WebSocket integration provides real-time updates for:
- Analysis progress tracking
- Batch job status updates
- System alerts and notifications
- Live dashboard metrics

## UI Components

### Design System
- Consistent color palette with primary, success, warning, and error variants
- Typography scale using Inter font
- Spacing and sizing utilities
- Responsive breakpoints

### Component Library
- **Forms** - Input fields, dropdowns, file uploads
- **Tables** - Sortable, filterable data tables
- **Charts** - Score visualizations, trend charts, progress indicators
- **Modals** - Confirmation dialogs, detail views
- **Notifications** - Toast messages, alert banners
- **Navigation** - Sidebar, breadcrumbs, tabs

## Performance Optimization

- **Code Splitting** - Route-based and component-based splitting
- **Lazy Loading** - Deferred loading of heavy components
- **Memoization** - React.memo and useMemo for expensive operations
- **Bundle Optimization** - Tree shaking and chunk optimization
- **Caching** - Aggressive caching with React Query

## Testing

```bash
# Run unit tests
npm run test

# Run with coverage
npm run test:coverage

# Run E2E tests
npm run test:e2e
```

## Deployment

### Build for Production

```bash
npm run build
```

### Static Hosting
The built application can be deployed to any static hosting service (Vercel, Netlify, AWS S3, etc.).

### Docker Deployment

```dockerfile
FROM node:18-alpine as builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API URL | `http://localhost:8000/api` |
| `VITE_WS_URL` | WebSocket URL | `ws://localhost:8000` |
| `VITE_APP_NAME` | Application name | `QuickScore` |
| `VITE_APP_VERSION` | Application version | `1.0.0` |

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Contributing

1. Follow the established code style and conventions
2. Write tests for new features
3. Update documentation as needed
4. Use semantic commit messages

## License

MIT License - see LICENSE file for details