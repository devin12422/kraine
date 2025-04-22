use num_traits::float::Float;
use std::collections::{BTreeMap};
use rustc_hash::{FxHashMap, FxHashSet};
use argmin::core::{Error, CostFunction, Gradient, Executor};
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use argmin::solver::quasinewton::LBFGS;
use nalgebra::Vector3;
use argmin_observer_slog::SlogLogger;
use std::mem;
use std::ops::ControlFlow;
use argmin::core::observers::ObserverMode;
use argmin::solver::linesearch::{MoreThuenteLineSearch,HagerZhangLineSearch};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict};
use pyo3::ffi::c_str;
use meshopt::{ generate_vertex_remap};
use pyo3::IntoPyObjectExt;
extern crate meshopt;


#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct DevelopArgs {
    #[arg(value_enum)]

    linesearch: LinesearchType,
    #[arg(short, long, value_name = "FILE")]
    save_as: Option<PathBuf>,
    #[arg(short, long, value_name = "FILE")]
    path: Option<PathBuf>,
    #[arg(short, long,default_value = "0")]
    mesh_index: usize,
    #[arg(short, long,default_value = "10")]
    iters: u64,
    #[arg(long,default_value = "1e-4")]
    c1: f64,
    #[arg(long,default_value = "0.9")]
    c2: f64,
    #[arg(long,default_value = "0.1")]
    delta: f64,
    #[arg(long,default_value = "0.9")]
    sigma: f64,
    #[arg(long,default_value = "0.66")]
    gamma: f64,
    #[arg(long,default_value = "0.01")]
    eta: f64,
    #[arg(long,default_value = "1e-6")]
    epsilon: f64,
    #[arg(long,default_value = "0.5")]
    theta: f64,
}
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum,Debug)]
enum LinesearchType {
    MoreThuente,
    HagerZhang
}
#[derive(Args,Copy, Clone,)]
struct MoreThuenteArgs {
    #[arg(long,default_value = "1e-4")]
    c1:f64,
    #[arg(long,default_value = "0.9")]
    c2:f64,
}
impl Default for MoreThuenteArgs {
    fn default() -> Self {
        MoreThuenteArgs{
            c1:1e-4, c2:0.9
        }
    }
}
#[derive(Args, Debug, Default, Clone)]
struct HagerZhangArgs {
}
#[derive(Clone, Debug)]
struct Partition{
    faces: FxHashSet<Face>,
    edges:[[u32;2];2],
    average_normal:Vector3<f64>,
    score:f64,
}

#[derive(PartialEq, Copy, Clone, Hash, Eq, Debug)]
struct Face([u32; 3]);
struct VertexPartition{
    partitions: [Option<Partition>; 2],
    score:f64
}
#[derive(Clone, Debug)]
struct MeshDevelopability {
    faces: Vec<Face>,
    vertex_face_map:BTreeMap<u32,Vec<Face>>,
}

impl Face{
    fn get_face_index_edges(&self,index:usize) ->[[u32;2];2]{
        [[self.0[index],self.0[(index+1)%3]],[self.0[index],self.0[(index-1)%3]]]
    }
    fn get_face_index_face_edges(&self,index:usize) ->[[usize;2];2]{
        [[index,(index+1)%3],[index,(index +2 )%3]]
    }
    fn get_face_index_other_face_edge(&self,index:usize,edge:[usize;2]) ->[usize;2]{
        let index_edges = self.get_face_index_face_edges(index);
        if index_edges[0] == edge {
            index_edges[1]
        } else { index_edges[0] }
    }
    fn get_face_index_of_index(&self,vertex_index:u32) ->Option<usize>{
        for index in 0..3{
            if self.0[index] == vertex_index{
                return Some(index);
            }
        }
        println!("Could not find face_index ");
        None
    }
    fn convert_face_edge_to_edge(&self,face_edge:[usize;2])->[u32;2]{
        [self.0[face_edge[0]],self.0[face_edge[1]]]
    }
    fn cmp_edge_and_face_edge(&self,edge:[u32;2],face_edge:[usize;2]) ->bool{

        edge == self.convert_face_edge_to_edge(face_edge)
    }
    fn get_face_index_edge_other_edge(&self,index:usize,edge:[u32;2]) ->[u32;2]{
        let face_edges = self.get_face_index_face_edges(index);
        if self.cmp_edge_and_face_edge(edge, face_edges[0]) {
            self.convert_face_edge_to_edge(face_edges[1])
        } else {
            self.convert_face_edge_to_edge(face_edges[0])
        }
    }
}

#[inline]
fn get_vertex_position(positions:&[f64],vertex_index:u32)->Vector3<f64>{
    Vector3::from([positions[vertex_index as usize],positions[vertex_index as usize +1],positions[vertex_index as usize+2]])
}
impl MeshDevelopability {
    fn get_edge_adjacent_faces(&self, edge:[u32;2]) ->Vec<&Face>{
        let mut adjacent_faces = Vec::new();
        for face in&self.vertex_face_map[&edge[0]]{
            for rhs in &self.vertex_face_map[&edge[1]]{
                if rhs == face{
                    adjacent_faces.push(face);
                }
            }
        }
        adjacent_faces
    }
    fn get_other_side_of_edge(&self, face:Face,edge:[u32;2]) ->Option<&Face>{
        let faces = self.get_edge_adjacent_faces(edge);
        if faces.len() <= 1{
            return None
        }
        if faces[0] == &face {
            Some(faces[1])
        } else { Some(faces[0]) }
    }
    #[inline]
    fn calculate_face_normal(&self,face:&Face,p:&[f64])->Vector3<f64>{
        let i = get_vertex_position(p,face.0[0]);
        (get_vertex_position(p,face.0[1])- i).cross(&(i-get_vertex_position(p,face.0[2])))
    }

    fn create_vertex_partition_map(&self, p: &[f64]) -> FxHashMap<u32,VertexPartition>{
        let mut vertex_partition_map:FxHashMap<u32,VertexPartition> =FxHashMap::default();
        for partition_face in &self.faces{
            'vertex_star_loop : for vertex_star_face_index in 0..3 {
                let vertex_star_index = partition_face.0[vertex_star_face_index];
                let vertex_star_faces = self.vertex_face_map.get(&vertex_star_index).unwrap();
                let vertex_star_arity = vertex_star_faces.len();
                if vertex_star_arity <= 3{
                    continue 'vertex_star_loop;
                }else if let ControlFlow::Continue(vertex_partition) = (1..=vertex_star_arity).try_fold(VertexPartition{partitions:[None,None],score:f64::MAX},| mut vertex_partition,big_partition_size|{
                   // The primary partition contains at least one face in a given vertex star, but can contain all of them
                    // For all faces, we iterate over both edges.
                    // There is no mechanism to determine if an edge has been used for partitioning previously.
                    // This doubles the amount of work done per triangle in a given mesh
                    //I believe we can get away with iterating over just 1 edge, but this code lets be lazy AND correct  *hopefully
                    let partition_face_edges = partition_face.get_face_index_face_edges(vertex_star_face_index);
                    if let ControlFlow::Continue(current_partition) = partition_face_edges.iter().try_fold(VertexPartition{partitions:[None,None],score:f64::MAX},|mut vertex_partition,partition_face_edge|{
                       let partition_edge = partition_face.convert_face_edge_to_edge(*partition_face_edge);
                       if let ControlFlow::Continue(current_partition) = (0..2).try_fold(VertexPartition{partitions:[None,None],score:0.0f64},|mut vertex_partition,i|{
                           let partition_size = if i == 0 {big_partition_size} else {vertex_star_arity - big_partition_size};
                           if partition_size == 0{
                               ControlFlow::Continue(vertex_partition)// We do not iterate over the second partition if it's size is 0 (this may be the second partition)
                           }else
                           {
                               let mut partition = Partition{
                                   faces: FxHashSet::default(),
                                   edges: [partition_edge,partition_edge],
                                   average_normal: Vector3::<f64>::zeros(),
                                   score: 0.0,
                               };
                              //This should change the iteration direction and partition starting face based off of which partition from the set we're currently calculating (i)
                               let face = if i == 0
                               {
                                   Some(partition_face.clone())
                               } else
                               {
                                   self.get_other_side_of_edge(*partition_face,partition_edge).cloned()
                               };
                               if let Some(face) = face{
                                   // We now have our partition parameters (starting edge, partition size, and iteration direction
                                   // We use a different iterator from the others here, and we have to do the same iteration twice
                                   if (0..partition_size).try_fold(face, |face, _index|{
                                       // partition.average_normal += self.calculate_face_normal(&face,p);
                                       partition.faces.insert(face);
                                       //Finish up iteration
                                       partition.edges[1] = face.get_face_index_edge_other_edge(face.get_face_index_of_index(vertex_star_index).unwrap(), partition.edges[1]);
                                       self.get_other_side_of_edge(face, partition.edges[1]).copied()
                                   }).is_some()
                                   {
                                       // partition.average_normal /= partition.faces.len() as f64; // DO NOT DIVIDE BY VERTEX_ARITY
                                       // DIVIDE BY PARTITION SIZE
                                       for face in &partition.faces{
                                           for other_face in &&partition.faces{
                                               if other_face != face{
                                                   let score = (self.calculate_face_normal(face,p) - self.calculate_face_normal(other_face,p)).magnitude();
                                                   // let score = (self.calculate_face_normal(face,p).normalize()-partition.average_normal.normalize()).magnitude_squared();
                                                   if score > partition.score {
                                                       // The score for a given partition is the maximum of
                                                       // the magnitude squared of a face's normal from the average
                                                       // for all of it's faces
                                                       partition.score = score;
                                                   }
                                               }

                                           }

                                       }
                                       if partition.score > vertex_partition.score {
                                           //The score of a vertex partition (pair of partitions)
                                           // is the maximum score of it's two partitions
                                           vertex_partition.score = partition.score;
                                       }
                                       if vertex_partition.partitions[0].is_none(){
                                           vertex_partition.partitions[0] = Some(partition);

                                       }else{
                                           vertex_partition.partitions[1] = Some(partition);
                                       }
                                       ControlFlow::Continue(vertex_partition)
                                   } else
                                   {
                                       ControlFlow::Break(vertex_partition) //Interior iteration has failed, return None
                                   }
                               }else{
                                   ControlFlow::Break(vertex_partition)
                               }
                           }

                       }){
                           if vertex_partition.score > current_partition.score{
                               vertex_partition = current_partition;
                           }
                           ControlFlow::Continue(vertex_partition)
                       }else{
                           ControlFlow::Break(vertex_partition)
                       }

                   }){
                        if vertex_partition.score > current_partition.score{
                            vertex_partition = current_partition;
                        }
                        ControlFlow::Continue(vertex_partition)
                    }else{
                        ControlFlow::Break(vertex_partition)
                    }

                }){
                   // If out of all the partition sizes,
                   // expanding out in either direction, starting from this face,
                   // we have obtained the partition set with the lowest score
                    vertex_partition_map.insert(vertex_star_index, vertex_partition);
                }
            }
        }
        vertex_partition_map
    }
}
impl CostFunction for MeshDevelopability {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {

        let partition_map = self.create_vertex_partition_map(p);
        let mut sum = 0f64;
        for vertex_partition in partition_map.values(){
            //println!("Score: {}",vertex_partition.score);
            sum += vertex_partition.score;
        }
        if sum.is_nan() || sum == 0f64{
            sum = f64::MAX;
        }
        //println!("Score: {:?}",sum);
        Ok(sum)
    }
}

impl Gradient for MeshDevelopability {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        let partition_map = self.create_vertex_partition_map(p);
        let mut gradients = vec![0f64;p.len()];
        for (i_index,vertex_partition) in partition_map.iter(){
            for partition in vertex_partition.partitions.iter().flatten(){
                    for face in partition.faces.iter(){
                        for other_face in partition.faces.iter(){
                            if face != other_face{
                                let i1 = get_vertex_position(p,face.0[0]);
                                let i2 = get_vertex_position(p,other_face.0[0]);
                                let i1_face_index = face.get_face_index_of_index(i_index.clone()).unwrap();
                                let i2_face_index = other_face.get_face_index_of_index(i_index.clone()).unwrap();
                                // Do we want to iterate using zeroes or relative to face indexes
                                let face1_normal = (get_vertex_position(p,face.0[1])- i1).cross(&(i-get_vertex_position(p,face.0[2])));
                                let face2_normal = (get_vertex_position(p,other_face.0[1])- i2).cross(&(i-get_vertex_position(p,face.0[2])));
                                let j1_index = face.0[(i1_face_index +1)%3];
                                let j2_index = other_face.0[(i2_face_index +1)%3];

                                let k1_index = face.0[(i1_face_index +2)%3];
                                let k2_index = other_face.0[(i2_face_index +2)%3];

                                let i = get_vertex_position(p,face.0[i_face_index]);
                                let j = get_vertex_position(p,j_index);
                                let k = get_vertex_position(p,k_index);
                                let i2 = get_vertex_position(p,face.0[i_face_index]);
                                let j2 = get_vertex_position(p,j2_index);
                                let k2 = get_vertex_position(p,k2_index);
                                let ejk = k - j;
                                let eki = i-k;
                                let eij = j- i;
                                let lhs = face_normal - partition.average_normal;
                                let area = face_normal.norm()/2f64;
                                let mut d_ndi = ejk.cross(&face_normal)*face_normal.transpose()/ area;
                                let mut d_ndj = eki.cross(&face_normal)*face_normal.transpose()/ area;
                                let mut d_ndk = eij.cross(&face_normal)*face_normal.transpose()/ area;
                                d_ndi -= d_ndi.unscale(partition.faces.len() as f64);
                                d_ndj -= d_ndj.unscale(partition.faces.len() as f64);
                                d_ndk -= d_ndk.unscale(partition.faces.len() as f64);
                                let g_ndi = lhs.transpose()* d_ndi;
                                let g_ndj = lhs.transpose()* d_ndj;
                                let g_ndk = lhs.transpose()* d_ndk;
                                for i in 0usize..3usize{
                                    gradients[*i_index as usize + i] += g_ndi.data.0[i][0];
                                    gradients[k_index as usize + i] += g_ndk.data.0[i][0];
                                    gradients[j_index as usize+ i] += g_ndj.data.0[i][0];
                                }
                            }
                        }
                        // if true{
                        //         let mut normal_gradient:Matrix3<f32> =  ((i-j).cross(&face_normal))* face_normal.transpose().unscale(area);
                        //         normal_gradient -= normal_gradient / partition.faces.len() as f32;
                        //         let index = face.0[face_index] as usize;
                        //         let gradient:Vector3<f32> =  (lhs.transpose()* normal_gradient).transpose().scale(2f32);
                        //         gradients[index] += gradient[0];
                        //         gradients[index+1] += gradient[1];
                        //         gradients[index+2] += gradient[2];
                            // }else{
                            //     let grad_area = face_normal.unscale(2f32).cross(&(i-j)).pseudo_inverse(f32::default_epsilon()).unwrap();
                            //     let mut normal_gradient =( grad_area * (i-j).cross(&face_normal)) * face_normal.transpose();
                            //     normal_gradient -= normal_gradient / partition.faces.len() as f32;
                            //     let index = face.0[face_index] as usize;
                            //     let gradient =  (lhs.dot(&normal_gradient)).scale(2f32);
                            //     gradients[index] += gradient[0];
                            //     gradients[index+1] += gradient[1];
                            //     gradients[index+2] += gradient[2];
                            // };
                }
            }
        }
        Ok(gradients)
    }
}
fn main() {
    let args = DevelopArgs::parse();
    let input_mesh_path = args.path.unwrap_or(rfd::FileDialog::new().add_filter("OBJ",&["obj"]).pick_file().unwrap());
    let (models, _) = tobj::load_obj(
        &input_mesh_path,
        &tobj::LoadOptions {
            ignore_lines:true,
            ignore_points:true,
            triangulate: true,
            ..Default::default()
        },
    ).unwrap();

    let loaded_mesh = models[args.mesh_index].clone().mesh;

    let indices:Vec<u32> = vec![0;loaded_mesh.indices.len()];
    let (total_vertices, vertex_remap) = generate_vertex_remap(&loaded_mesh.positions, Some(&loaded_mesh.indices));
    println!("Total vertices: {}",total_vertices);

    unsafe {
        meshopt::ffi::meshopt_remapIndexBuffer(
            indices.as_ptr() as *mut ::std::os::raw::c_uint,
            loaded_mesh.indices.as_ptr() as *mut ::std::os::raw::c_uint,
            loaded_mesh.indices.len(),
            vertex_remap.as_ptr() as *const ::std::os::raw::c_uint,
        );
    }
    println!("Total indices: {}",indices.len());
    let vertices:Vec<[f32;3]> =vec![[0f32,0f32,0f32];total_vertices];
    unsafe {
        meshopt::ffi::meshopt_remapVertexBuffer(
            vertices.as_ptr() as *mut ::std::os::raw::c_void,
            loaded_mesh.positions.as_ptr() as *const ::std::os::raw::c_void,
            total_vertices,
            mem::size_of::<[f32;3]>(),
            vertex_remap.as_ptr() as *const ::std::os::raw::c_uint,
        );
    }
    let mut f64_vertices = Vec::new();
    for vertex in vertices{
        f64_vertices.push([vertex[0] as f64,vertex[1]as f64,vertex[2]as f64]);
    }
    let mut triangles: Vec<[u32;3]> = Vec::new();
    let mut vertex_face_map: BTreeMap<u32,Vec<Face>> = BTreeMap::new();
    let faces = (0..indices.len()/3).map(|face_index|{
        let face = Face([indices[face_index*3],indices[face_index*3+1],indices[face_index*3+2]]);
        triangles.push(face.0);
        for vertex_index in face.0{
            if let std::collections::btree_map::Entry::Vacant(e) = vertex_face_map.entry(vertex_index) {
                e.insert(vec![face]);
            } else {
                vertex_face_map.get_mut(&vertex_index).unwrap().push(face);
            }
        }
        face
    }).collect::<Vec<Face>>();
    println!("Total triangles: {}",triangles.len());

    Python::with_gil(|py| {
        let developability_problem = MeshDevelopability{faces,vertex_face_map};
        // let linesearch = argmin::solver::linesearch::HagerZhangLineSearch::new();
            //.with_delta_sigma(0.1,0.2)
            //.unwrap();
         let res_vertices = match args.linesearch{
            LinesearchType::MoreThuente => {
                let linesearch =MoreThuenteLineSearch::new().with_c(args.c1,args.c2).unwrap();
                let solver = LBFGS::new(linesearch, 7).with_tolerance_cost(f64::epsilon()).unwrap();
                Executor::new(developability_problem.clone(), solver)
                    .configure(|state| state.param(f64_vertices.into_flattened()).max_iters(args.iters))
                    .add_observer(SlogLogger::term_noblock(), ObserverMode::NewBest)
                    .run().unwrap().state.best_param.unwrap()
            }
            LinesearchType::HagerZhang => {
                let linesearch =argmin::solver::linesearch::HagerZhangLineSearch::new().with_delta_sigma(args.delta,args.sigma).unwrap().with_epsilon(args.epsilon).unwrap().with_gamma(args.gamma).unwrap().with_theta(args.theta).unwrap().with_eta(args.eta).unwrap();
                let solver = LBFGS::new(linesearch, 7).with_tolerance_cost(f64::epsilon()).unwrap();
                Executor::new(developability_problem.clone(), solver)
                    .configure(|state| state.param(f64_vertices.into_flattened()).max_iters(args.iters))
                    .add_observer(SlogLogger::term_noblock(), ObserverMode::NewBest)
                    .run().unwrap().state.best_param.unwrap()
            }
        };
        let mut grad:Vec<f64> = vec![0f64;total_vertices];
        grad.resize(total_vertices,0f64);
        let mut developed_vertices = Vec::new();
        let partition_map = developability_problem.create_vertex_partition_map(&res_vertices);
        for (index,partition) in partition_map{
            grad[index as usize] = partition.score;
        }
        for index in 0..total_vertices {
            developed_vertices.push([res_vertices[index*3] as f32, res_vertices[index*3+1] as f32,res_vertices[index*3+2] as f32]);
        };

        let ps = py.import("polyscope").unwrap();
        let np = py.import("numpy").unwrap();
        let np_array = np.getattr("array").unwrap();
        let py_vertices = np_array.call1(( developed_vertices.into_bound_py_any(py).unwrap(),)).unwrap();
        let py_indices =np_array.call1(( triangles.into_bound_py_any(py).unwrap(),)).unwrap();
        let py_grad =  np_array.call1((grad.into_bound_py_any(py).unwrap(),)).unwrap();

        let locals = [("ps", py.import("polyscope").unwrap()),].into_py_dict(py).unwrap();
        // ("verts",py_vertices),("faces",py_indices)
        py.eval( c_str!("ps.init()"), None, Some(&locals)).unwrap();

        ps.getattr("register_surface_mesh").unwrap().call1(("my_mesh",py_vertices,py_indices)).unwrap();
        let kwargs = [("defined_on", "vertices"),("cmap","blues")].into_py_dict(py).unwrap();
        py.eval( c_str!("ps.get_surface_mesh('my_mesh')"), None, Some(&locals)).unwrap().getattr("add_scalar_quantity").unwrap().call(("grad",py_grad,),Some(&kwargs)).unwrap();


        py.eval( c_str!("ps.show()"), None, Some(&locals)).unwrap();

    });
    // Print result
}
