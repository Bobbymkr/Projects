import React from 'react';
import { Card, Skeleton, Row, Col, Space } from 'antd';

interface MetricCardSkeletonProps {
  count?: number;
}

export const MetricCardSkeleton: React.FC<MetricCardSkeletonProps> = ({ count = 4 }) => {
  return (
    <Row gutter={[24, 24]}>
      {Array.from({ length: count }).map((_, index) => (
        <Col xs={24} sm={12} md={8} lg={6} key={index}>
          <Card className="metric-card">
            <Skeleton active paragraph={{ rows: 2 }} title={{ width: '60%' }} />
          </Card>
        </Col>
      ))}
    </Row>
  );
};

interface ChartSkeletonProps {
  height?: number;
  title?: string;
}

export const ChartSkeleton: React.FC<ChartSkeletonProps> = ({ height = 300, title }) => {
  return (
    <Card title={title} className="chart-container">
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Skeleton.Node active style={{ width: '100%', height: height - 48 }}>
          <div style={{ width: '100%', height: '100%' }} />
        </Skeleton.Node>
      </div>
    </Card>
  );
};

interface TableSkeletonProps {
  rows?: number;
  columns?: number;
}

export const TableSkeleton: React.FC<TableSkeletonProps> = ({ rows = 5, columns = 4 }) => {
  return (
    <Card className="chart-container">
      <Skeleton active paragraph={{ rows: 1 }} title={{ width: '40%' }} />
      <div style={{ marginTop: 24 }}>
        {Array.from({ length: rows }).map((_, rowIndex) => (
          <Row gutter={16} key={rowIndex} style={{ marginBottom: 16 }}>
            {Array.from({ length: columns }).map((_, colIndex) => (
              <Col span={24 / columns} key={colIndex}>
                <Skeleton.Input active size="small" style={{ width: '100%' }} />
              </Col>
            ))}
          </Row>
        ))}
      </div>
    </Card>
  );
};

interface MapSkeletonProps {
  height?: number;
}

export const MapSkeleton: React.FC<MapSkeletonProps> = ({ height = 600 }) => {
  return (
    <Card title="Traffic Operations Center" className="map-container">
      <div 
        style={{ 
          height, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          background: '#f0f8ff',
          borderRadius: 8,
        }}
      >
        <Space direction="vertical" align="center">
          <Skeleton.Avatar active size={64} shape="circle" />
          <Skeleton active paragraph={{ rows: 2, width: ['80%', '60%'] }} title={false} />
        </Space>
      </div>
    </Card>
  );
};

interface DashboardSkeletonProps {
  type?: 'executive' | 'operations' | 'analytics' | 'system';
}

export const DashboardSkeleton: React.FC<DashboardSkeletonProps> = ({ type = 'executive' }) => {
  switch (type) {
    case 'executive':
      return (
        <div className="executive-dashboard">
          <Skeleton active paragraph={{ rows: 0 }} style={{ marginBottom: 24 }} />
          <MetricCardSkeleton count={4} />
          <div style={{ marginTop: 24 }}>
            <ChartSkeleton height={200} title="Financial & Environmental Impact" />
          </div>
          <Row gutter={[24, 24]} style={{ marginTop: 24 }}>
            <Col xs={24} sm={12} md={8}>
              <Card title="System Health" className="metric-card">
                <Skeleton active paragraph={{ rows: 4 }} />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={8}>
              <Card title="User Satisfaction" className="metric-card">
                <Skeleton active paragraph={{ rows: 3 }} />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={8}>
              <Card title="Alert Summary" className="metric-card">
                <Skeleton active paragraph={{ rows: 3 }} />
              </Card>
            </Col>
          </Row>
        </div>
      );

    case 'operations':
      return (
        <div className="operations-dashboard">
          <Skeleton active paragraph={{ rows: 0 }} style={{ marginBottom: 24 }} />
          <Row gutter={[24, 24]}>
            <Col xs={24} lg={16}>
              <MapSkeleton height={600} />
            </Col>
            <Col xs={24} lg={8}>
              <Space direction="vertical" size="large" style={{ width: '100%' }}>
                <Card title="Intersection Control" className="control-panel">
                  <Skeleton active paragraph={{ rows: 6 }} />
                </Card>
                <Card title="System Status" className="status-panel">
                  <Skeleton active paragraph={{ rows: 4 }} />
                </Card>
              </Space>
            </Col>
          </Row>
        </div>
      );

    case 'analytics':
      return (
        <div className="analytics-dashboard">
          <Skeleton active paragraph={{ rows: 0 }} style={{ marginBottom: 24 }} />
          <TableSkeleton rows={5} columns={4} />
          <Row gutter={[24, 24]} style={{ marginTop: 24 }}>
            <Col xs={24} lg={12}>
              <ChartSkeleton height={300} title="Traffic Volume Patterns" />
            </Col>
            <Col xs={24} lg={12}>
              <Card title="Algorithm Efficiency" className="chart-container">
                <Skeleton active paragraph={{ rows: 6 }} />
              </Card>
            </Col>
          </Row>
        </div>
      );

    case 'system':
      return (
        <div className="system-status">
          <Skeleton active paragraph={{ rows: 0 }} style={{ marginBottom: 24 }} />
          <MetricCardSkeleton count={6} />
          <Row gutter={[24, 24]} style={{ marginTop: 24 }}>
            <Col xs={24} sm={12} md={8}>
              <Card title="System Information" className="info-card">
                <Skeleton active paragraph={{ rows: 4 }} />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={8}>
              <Card title="Service Status" className="service-card">
                <Skeleton active paragraph={{ rows: 6 }} />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={8}>
              <Card title="Recent Alerts" className="alerts-card">
                <Skeleton active paragraph={{ rows: 4 }} />
              </Card>
            </Col>
          </Row>
        </div>
      );

    default:
      return <Skeleton active />;
  }
};

export default DashboardSkeleton;

