# 项目初始化脚本
#!/bin/bash

echo "🚀 开始创建 RAG 系统项目结构..."

## 创建根目录
#mkdir -p rag-system
#cd rag-system

# 创建主要模块目录
mkdir -p {rag-engine,rag-api,rag-frontend,infrastructure}

echo "📁 创建 Rust 引擎项目..."
cd rag-engine || exit

# 初始化 Rust 项目
cargo init --name rag-engine

# 创建 Rust 项目目录结构
mkdir -p src/{config,error,types,utils,concurrency,cache,pipeline,embedding,retrieval,llm,multimodal,plugins,observability,controllers,services,network}
mkdir -p src/pipeline/{ingestion,preprocessing,versioning}
mkdir -p src/embedding/{providers,models}
mkdir -p src/retrieval/{stores,strategies,fusion,reranking,query_processing}
mkdir -p src/llm/{providers,prompts,conversation,generation}
mkdir -p src/multimodal/{vision,audio,video,generation}
mkdir -p src/plugins/{runtime,registry,interfaces,builtin}
mkdir -p src/observability/{metrics,tracing,logging,health,evaluation}
mkdir -p src/network/{grpc,http,websocket}

mkdir -p {proto,tests,benches,examples,docs}
mkdir -p tests/{integration,performance,e2e}

echo "🦀 配置 Rust Cargo.toml..."

# 返回根目录
cd ..

echo "🐹 创建 Go API 项目..."
cd rag-api || exit

# 初始化 Go 模块
go mod init rag-api

# 创建 Go 项目目录结构
mkdir -p cmd/{server,worker,cli}
mkdir -p internal/{config,gateway,handlers,services,clients,models,repositories,workers,queues,utils,auth,monitoring,errors}
mkdir -p pkg/{grpc,http,cache,database,logger}
mkdir -p api/{openapi,grpc}
mkdir -p {migrations,configs,deployments,scripts,tests}
mkdir -p deployments/{docker,kubernetes,helm}
mkdir -p tests/{integration,e2e,load}

cd ..

echo "⚛️ 创建 React 前端项目..."
cd rag-frontend || exit

# 使用 Vite 创建 React 项目
npm create vite@latest . -- --template react-ts

# 创建前端目录结构
mkdir -p src/{routes,pages,components,store,services,hooks,utils,types,styles,config}
mkdir -p src/pages/{auth,dashboard,documents,search,chat,workspaces,admin,profile,error}
mkdir -p src/components/{layout,ui,business,features}
mkdir -p src/store/{slices,selectors}
mkdir -p src/services/{api,websocket,storage,analytics,workers}
mkdir -p src/hooks/{auth,data,ui,websocket,performance}
mkdir -p src/utils/{constants,helpers,api,performance}
mkdir -p src/types/{api,ui,store,global}
mkdir -p src/styles/{themes,components,pages}

cd ..

echo "🏗️ 创建基础设施目录..."
cd infrastructure || exit

mkdir -p {docker,kubernetes,helm,terraform,monitoring,scripts,ci-cd,security,docs}
mkdir -p docker/{rag-engine,rag-api,rag-frontend,docker-compose}
mkdir -p kubernetes/{namespaces,configs,secrets,storage,deployments,services,ingress,hpa,network-policies}
mkdir -p helm/{rag-system,monitoring}
mkdir -p terraform/{environments,modules,shared}
mkdir -p monitoring/{prometheus,grafana,jaeger,loki,alertmanager}
mkdir -p scripts/{build,deploy,database,monitoring,utilities}
mkdir -p ci-cd/{github-actions,gitlab-ci,jenkins}
mkdir -p security/{network-policies,rbac,pod-security,certificates}
mkdir -p docs/{architecture,deployment,development,operations}

cd ..

echo "✅ 项目结构创建完成！"
echo "📊 项目概览："
tree -L 2 -d .

echo ""
echo "🎯 下一步："
echo "1. 配置 Rust 依赖 (Cargo.toml)"
echo "2. 配置 Go 依赖 (go.mod)"
echo "3. 配置前端依赖 (package.json)"
echo "4. 设置 Docker 和开发环境"