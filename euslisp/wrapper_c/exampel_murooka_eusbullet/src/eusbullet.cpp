#include <btBulletCollisionCommon.h>
#include <BulletCollision/NarrowPhaseCollision/btGjkCollisionDescription.h>
#include <BulletCollision/NarrowPhaseCollision/btVoronoiSimplexSolver.h>
#include <BulletCollision/NarrowPhaseCollision/btComputeGjkEpaPenetration.h>
#include <LinearMath/btGeometryUtil.h>


struct btDistanceInfo
{ // this class is copied from https://github.com/bulletphysics/bullet3/blob/master/test/collision/btDistanceInfo.h
  btVector3 m_pointOnA;
  btVector3 m_pointOnB;
  btVector3 m_normalBtoA;
  btScalar m_distance;
};


struct ConvexWrap
{ // this class is copied from https://github.com/bulletphysics/bullet3/blob/master/test/collision/main.cpp
  btConvexShape* m_convex;
  btTransform m_worldTrans;
  inline btScalar getMargin() const
  {
    return m_convex->getMargin();
  }
  inline btVector3 getObjectCenterInWorld() const
  {
    return m_worldTrans.getOrigin();
  }
  inline const btTransform& getWorldTransform() const
  {
    return m_worldTrans;
  }
  inline btVector3 getLocalSupportWithMargin(const btVector3& dir) const
  {
    return m_convex->localGetSupportingVertex(dir);
  }
  inline btVector3 getLocalSupportWithoutMargin(const btVector3& dir) const
  {
    return m_convex->localGetSupportingVertexWithoutMargin(dir);
  }
};

long makeSphereModel(double radius)
{
  return (long)(new btSphereShape(radius));
};

long makeBoxModel(double xsize, double ysize, double zsize)
{
  return (long)(new btBoxShape(0.5*btVector3(xsize, ysize, zsize)));
};

long makeCylinderModel(double radius, double height)
{
  return (long)(new btCylinderShapeZ(btVector3(radius, radius, 0.5*height)));
};

long makeCapsuleModel(double radius, double height)
{
  return (long)(new btCapsuleShapeZ(radius, 0.5*height));
};

long makeMeshModel(double *verticesPoints, long numVertices)
{
  btConvexHullShape* pshape = new btConvexHullShape();
#define SHRINK_FOR_MARGIN false
  if (SHRINK_FOR_MARGIN) {
    // Shrink vertices for default margin CONVEX_DISTANCE_MARGIN,
    // which should be nonzero positive for fast computation of penetration distance.
    // ref: https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=2358#p9411
    // But sometimes, this doesn't work well (vertices become empty), so currently disabled.
    btAlignedObjectArray<btVector3> vertices;
    for (int i = 0; i < 3 * numVertices; i += 3) {
      vertices.push_back(btVector3(verticesPoints[i], verticesPoints[i+1], verticesPoints[i+2]));
    }
    btAlignedObjectArray<btVector3> planes;
    btGeometryUtil::getPlaneEquationsFromVertices(vertices, planes);
    int sz = planes.size();
    for (int i = 0 ; i < sz ; i++) {
      planes[i][3] += CONVEX_DISTANCE_MARGIN;
    }
    vertices.clear();
    btGeometryUtil::getVerticesFromPlaneEquations(planes, vertices);
    sz = vertices.size();
    for (int i = 0 ; i < sz ; i++) {
      pshape->addPoint(vertices[i]);
    }
  } else {
    for (int i = 0; i < 3 * numVertices; i += 3) {
      pshape->addPoint(btVector3(verticesPoints[i], verticesPoints[i+1], verticesPoints[i+2]));
    }
  }
  return (long)pshape;
};

long calcCollisionDistance(long modelAddrA, long modelAddrB,
                           double *posA, double *quatA, double *posB, double *quatB,
                           double *dist, double *dir, double *pA, double *pB)
{
  ConvexWrap a, b;
  a.m_convex = ((btConvexShape *)modelAddrA);
  a.m_worldTrans.setOrigin(btVector3(posA[0], posA[1], posA[2]));
  a.m_worldTrans.setRotation(btQuaternion(quatA[1], quatA[2], quatA[3], quatA[0])); // w is first element in euslisp
  b.m_convex = ((btConvexShape *)modelAddrB);
  b.m_worldTrans.setOrigin(btVector3(posB[0], posB[1], posB[2]));
  b.m_worldTrans.setRotation(btQuaternion(quatB[1], quatB[2], quatB[3], quatB[0])); // w is first element in euslisp
  // The origin of euslisp cylinder model is located on bottom, so local translation of half height is necessary
  if(btCylinderShapeZ* cly = dynamic_cast<btCylinderShapeZ*>(a.m_convex)) {
    btVector3 heightOffset(btVector3(0, 0, cly->getHalfExtentsWithMargin().getZ()));
    a.m_worldTrans.setOrigin(a.m_worldTrans.getOrigin() + a.m_worldTrans.getBasis() * heightOffset);
  }
  if(btCylinderShapeZ* cly = dynamic_cast<btCylinderShapeZ*>(b.m_convex)) {
    btVector3 heightOffset(btVector3(0, 0, cly->getHalfExtentsWithMargin().getZ()));
    b.m_worldTrans.setOrigin(b.m_worldTrans.getOrigin() + b.m_worldTrans.getBasis() * heightOffset);
  }

  btGjkCollisionDescription colDesc;
  btVoronoiSimplexSolver simplexSolver;
  btDistanceInfo distInfo;
  int res = -1;
  simplexSolver.reset();
  res = btComputeGjkEpaPenetration(a, b, colDesc, simplexSolver, &distInfo);

  // The result of btComputeGjkEpaPenetration is offseted by CONVEX_DISTANCE_MARGIN.
  // Although the offset is considered internally in primitive shapes, not considered in convex hull shape.
  // So, the result is modified manually.
  if(dynamic_cast<btConvexHullShape*>((btConvexShape *)modelAddrA)) {
    distInfo.m_distance += CONVEX_DISTANCE_MARGIN;
    distInfo.m_pointOnA += CONVEX_DISTANCE_MARGIN * distInfo.m_normalBtoA;
  }
  if(dynamic_cast<btConvexHullShape*>((btConvexShape *)modelAddrB)) {
    distInfo.m_distance += CONVEX_DISTANCE_MARGIN;
    distInfo.m_pointOnB += - CONVEX_DISTANCE_MARGIN * distInfo.m_normalBtoA;
  }

  *dist = distInfo.m_distance;
  for (int i = 0; i < 3; i++) {
    dir[i] = distInfo.m_normalBtoA[i];
    pA[i] = distInfo.m_pointOnA[i];
    pB[i] = distInfo.m_pointOnB[i];
  }

  return res;
};

long setMargin(long modelAddr, double margin)
{
  // shape are shrinked for CONVEX_DISTANCE_MARGIN, so CONVEX_DISTANCE_MARGIN is added to margin
  ((btConvexShape *)modelAddr)->setMargin(CONVEX_DISTANCE_MARGIN+margin);
  return 0;
};

extern "C" {
  long callMakeSphereModel(double radius)
  {
    return makeSphereModel(radius);
  }

  long callMakeBoxModel(double xsize, double ysize, double zsize)
  {
    return makeBoxModel(xsize, ysize, zsize);
  }

  long callMakeCylinderModel(double radius, double height)
  {
    return makeCylinderModel(radius, height);
  }

  long callMakeCapsuleModel(double radius, double height)
  {
    return makeCapsuleModel(radius, height);
  }

  long callMakeMeshModel(double *verticesPoints, long numVertices)
  {
    return makeMeshModel(verticesPoints, numVertices);
  }

  long callCalcCollisionDistance(long modelAddrA, long modelAddrB,
                                 double *posA, double *quatA, double *posB, double *quatB,
                                 double *dist, double *dir, double *pA, double *pB)
  {
    return calcCollisionDistance(modelAddrA, modelAddrB,
                                 posA, quatA, posB, quatB,
                                 dist, dir, pA, pB);
  }

  long callSetMargin(long modelAddr, double margin)
  {
    return setMargin(modelAddr, margin);
  }
}
