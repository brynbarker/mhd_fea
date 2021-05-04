#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/constrained_linear_operator.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;

template<int dim>
class MagneticField
{
    public:
        MagneticField() {}

        Tensor<1, dim> get_force(const Tensor<1,dim> &p) const
        {
            Tensor<1, dim> val;
            val[0] = 0.0;
            val[1] = 0.0;
            return val;
        }

};

template <int dim>
class NavierStokes
{
    public:
        NavierStokes(const unsigned int degree, 
                const unsigned int n_global_refinements);


        void run();

    private:
        void setup_grid();
        void setup_dofs();
        void assemble_system();
        void assemble_rhs();
        void output_results() const;

        const unsigned int degree;
        unsigned int n_global_refinements;

        double dt;
        double end_time;
        double current_time;
        int    time_step_number;

        double eta;
        double mu;
        double rho;

        MagneticField<dim> mag_field;

        Triangulation<dim> triangulation;
        FESystem<dim>      fe;
        DoFHandler<dim>    dof_handler;

        AffineConstraints<double> constraints;

        BlockSparsityPattern      sparsity_pattern;
        BlockSparsityPattern      unconstrained_sparsity_pattern;
        BlockSparseMatrix<double> system_matrix;
        //BlockSparseMatrix<double> mass_matrix;

        BlockSparseMatrix<double> unconstrained_system_matrix;
        BlockSparseMatrix<double> unconstrained_mass_matrix;

        BlockSparsityPattern      preconditioner_sparsity_pattern;
        BlockSparseMatrix<double> stiff_preconditioner;
        BlockSparseMatrix<double> mass_preconditioner;
        //SparseMatrix<double>      mass_u_matrix;
        SparseMatrix<double>      stiff_p_matrix;
        SparseMatrix<double>      mass_p_matrix;
        SparseMatrix<double>      mass_schur;

        BlockVector<double> current_solution;
        BlockVector<double> previous_solution;
        BlockVector<double> system_rhs;
        BlockVector<double> load_vector;
        BlockVector<double> convective_term;

        // maybe BlockSchurPreconditioner instead
        //std::shared_ptr< SparseDirectUMFPACK > A_preconditioner;
};

template<int dim>
class ExactSolution : public Function<dim>
{
    public:
        ExactSolution()
            : Function<dim>(dim+1) // 3
        {}

        virtual void
        vector_value(const Point<dim> &p,
                     Vector<double> &  values) const override
        {
            const double time = this->get_time();
            int order  = 0;

            if (order == 0)
            {
                values(0) = 1.0;
                values(1) = 1.0;
            }
            if (order == 1)
            {
                values(0) = -p(1);
                values(1) = p(0);
            }
            if (order == 2)
            {
                values(0) = -p(1)*p(1);
                values(1) = p(0)*p(0);
            }
            if (order == 3)
            {
                values(0) = time*cos(p(1));
                values(1) = time*sin(p(0));
            }
            values(2) = 0;
        }
};

template <int dim>
class ForcingFunction : public Function<dim>
{
    public:
        ForcingFunction(double eta, double mu, MagneticField<dim> &m_field)
            : Function<dim>(dim+1)
            , eta(eta)
            , mu(mu)
            , m_field(m_field)// 3
        {}

        virtual void
        vector_value(const Point<dim> &p,
                     Vector<double> &  values) const override
        {
            const double time = this->get_time();
            int order  = 0;

            if (order == 0)
            {
                values(0) = 0.0;
                values(1) = 0.0;
            }
            if (order == 1)
            {
                values(0) = -p(1);
                values(1) = p(0);
            }
            if (order == 2)
            {
                values(0) = -p(1)*p(1);
                values(1) = p(0)*p(0);
            }
            if (order == 3)
            {
                values(0) = time*cos(p(1));
                values(1) = time*sin(p(0));
            }
            values(2) = 0;
        }
    private:
        double eta;
        double mu;
        MagneticField<dim> &m_field;
};

class BlockSchurPreconditioner : public Subscriptor
{
    public:
        BlockSchurPreconditioner( double eta, 
                                  double mu,
                                  double dt,
                                  const BlockSparseMatrix<double> &system_matrix,
                                  //const SparseMatrix<double> &mass_u_matrix,
                                  const SparseMatrix<double> &stiff_p_matrix,
                                  const SparseMatrix<double> &mass_p_matrix,
                                  SparseMatrix<double> &mass_schur)
            : mu(mu)
            , eta(eta)
            , dt(dt)
            , system_matrix(&system_matrix)
            //, mass_u_matrix(&mass_u_matrix)
            , stiff_p_matrix(&stiff_p_matrix)
            , mass_p_matrix(&mass_p_matrix)
            , mass_schur(&mass_schur)
        {
            // define A inverse 
            A_inv.factorize(system_matrix.block(0,0));

            /*x
            // create storage
            Vector<double> tmp1, tmp2;
            tmp1.reinit(system_matrix.block(0,0).m());
            tmp2.reinit(system_matrix.block(0,0).m());
            tmp1 = 1.0;
            tmp2 = 0.0;

            // mass_schur = B(diag M_u)-1 BT
            PreconditionJacobi<> jacobi;
            //jacobi.initialize(mass_u_matrix);  // (diag M_u)^-1
            jacobi.vmult(tmp2, tmp1); // store as vector
            system_matrix.block(1,0).mmult(mass_schur,
                    system_matrix.block(0,1), tmp2);
            std::cout << "okay we made mass_schur\n";
            mass_schur_inv.factorize(mass_schur); // mass schur inv
            std::cout << "and facotirzed it\n";
            */

            std::cout << "here we are\n";
            // preconditioner for inverse pressure mass matrix
            preconditioner_Mp.initialize(mass_p_matrix);
            preconditioner_Kp.initialize(stiff_p_matrix);
            std::cout << "last line of construction\n";
        }

        void vmult(BlockVector<double> &sol, const BlockVector<double> &vec) const
        {
            // temp storage
            Vector<double> utmp(vec.block(0));
            Vector<double> tmp(vec.block(1));
            {
                // compute mass pressure inverse * vec(1)
                SolverControl solver_control(1000, 1e-6*vec.l2_norm());
                SolverCG<Vector<double>> cg(solver_control);
                sol.block(1) = 0.0;
                cg.solve(*mass_p_matrix, 
                         sol.block(1), 
                         vec.block(1), 
                         preconditioner_Mp);
                sol.block(1) *= -1.0*dt*(eta+mu+mu);
                
                // compute stiffness pressure inv * vec(1)
                //SolverControl solver_control(1000, 1e-6*vec.l2_norm());
                //SolverCG<Vector<double>> cg(solver_control);
                //sol.block(1) = 0.0;
                cg.solve(*stiff_p_matrix, 
                         tmp, 
                         vec.block(1), 
                         preconditioner_Kp);
                //mass_schur_inv.vmult(tmp, vec.block(1));
                tmp *= -1.0;
                sol.block(1) += tmp;
            }

            {
                // compute vec(0) - BT Sinv vec(1)
                system_matrix->block(0,1).vmult(utmp, sol.block(1));
                utmp *= -1.0;
                utmp += vec.block(0);

            }
            A_inv.vmult(sol.block(0), utmp);
            
        }
    private:

        const double mu;
        const double eta;
        const double dt;

        SparseDirectUMFPACK A_inv;
        SparseDirectUMFPACK mass_schur_inv;

        PreconditionSSOR<SparseMatrix<double>> preconditioner_Mp;
        PreconditionSSOR<SparseMatrix<double>> preconditioner_Kp;
        const SmartPointer<const BlockSparseMatrix<double> > system_matrix;
        //const SmartPointer<const SparseMatrix<double> > mass_u_matrix;
        const SmartPointer<const SparseMatrix<double> > stiff_p_matrix;
        const SmartPointer<const SparseMatrix<double> > mass_p_matrix;
        const SmartPointer<SparseMatrix<double> > mass_schur;

};

                


template <int dim>
NavierStokes<dim>::NavierStokes(const unsigned int degree,
                                const unsigned int n_global_refinements)
    : degree(degree)
    , n_global_refinements(n_global_refinements)
    , dt(std::pow(0.001, n_global_refinements))
    , current_time(0.0)
    , time_step_number(1)
    , eta(0.1)
    , mu(0.001)
    , rho(1.0)
    , fe(FE_Q<dim>(degree+1), dim, FE_Q<dim>(degree), 1)
    , dof_handler(triangulation)
{}

template <int dim>
void
NavierStokes<dim>::setup_grid()
{
  GridGenerator::hyper_cube(triangulation);

  //std::vector<GridTools::PeriodicFacePair<
  //    typename parallel::distributed::Triangulation<dim>::cell_iterator>>
  //    periodicity_vector;
  //GridTools::collect_periodic_faces(triangulation,
  //                                   2,
  //                                    3,
  //                                    1,
  //                                    periodicity_vector,
  //                                    Tensor<1, dim>(),
  //                                    rotation_matrix);

  triangulation.refine_global(n_global_refinements);
}


template <int dim>
void NavierStokes<dim>::setup_dofs()
{
    system_matrix.clear();
    unconstrained_mass_matrix.clear();
    unconstrained_system_matrix.clear();
    stiff_preconditioner.clear();
    mass_preconditioner.clear();
    //mass_u_matrix.clear();
    stiff_p_matrix.clear();
    mass_p_matrix.clear();

    dof_handler.distribute_dofs(fe);

    DoFRenumbering::Cuthill_McKee(dof_handler);
    // group together the velocity components 
    // separate from the pressure
    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;

    DoFRenumbering::component_wise(dof_handler, block_component);

    std::vector<types::global_dof_index> dofs_per_block(
            2, types::global_dof_index(0));

    DoFTools::count_dofs_per_block(dof_handler,
                                   dofs_per_block,
                                   block_component);

    const unsigned int n_u = dofs_per_block[0];
    const unsigned int n_p = dofs_per_block[1];

    // overkill to use exact solution here
    ExactSolution<dim> exact_solution;
    exact_solution.set_time(current_time);

    {
      constraints.clear();
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);

      // velocity and pressure boundary condition.
      VectorTools::interpolate_boundary_values(
          dof_handler,
          0,
          exact_solution,
          constraints);

      //FEValuesExtractors::Scalar q(dim);
    }

    constraints.close();

    {
        BlockDynamicSparsityPattern dsp(2, 2);
        dsp.block(0, 0).reinit(n_u, n_u);
        dsp.block(1, 0).reinit(n_p, n_u);
        dsp.block(0, 1).reinit(n_u, n_p);
        dsp.block(1, 1).reinit(n_p, n_p);

        dsp.collect_sizes();

        Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);

        for (unsigned int c = 0; c < dim + 1; ++c)
          for (unsigned int d = 0; d < dim + 1; ++d)
            if (!((c == dim) && (d == dim)))
              coupling[c][d] = DoFTools::always;
            else
              coupling[c][d] = DoFTools::none;

        DoFTools::make_sparsity_pattern(
          dof_handler, coupling, dsp, constraints, false);

        sparsity_pattern.copy_from(dsp);
    }
    
    {
        BlockDynamicSparsityPattern unconstrained_dsp(2, 2);
        unconstrained_dsp.block(0, 0).reinit(n_u, n_u);
        unconstrained_dsp.block(1, 0).reinit(n_p, n_u);
        unconstrained_dsp.block(0, 1).reinit(n_u, n_p);
        unconstrained_dsp.block(1, 1).reinit(n_p, n_p);

        unconstrained_dsp.collect_sizes();

        Table<2, DoFTools::Coupling> unconstrained_coupling(dim + 1, dim + 1);

        for (unsigned int c = 0; c < dim + 1; ++c)
          for (unsigned int d = 0; d < dim + 1; ++d)
            if (!((c == dim) && (d == dim)))
              unconstrained_coupling[c][d] = DoFTools::always;
            else
              unconstrained_coupling[c][d] = DoFTools::none;

        DoFTools::make_sparsity_pattern(dof_handler, 
                                        unconstrained_coupling, 
                                        unconstrained_dsp);

        unconstrained_sparsity_pattern.copy_from(unconstrained_dsp);
    }
    
    
    {
        BlockDynamicSparsityPattern preconditioner_dsp(2, 2);
        preconditioner_dsp.block(0, 0).reinit(n_u, n_u);
        preconditioner_dsp.block(1, 0).reinit(n_p, n_u);
        preconditioner_dsp.block(0, 1).reinit(n_u, n_p);
        preconditioner_dsp.block(1, 1).reinit(n_p, n_p);

        preconditioner_dsp.collect_sizes();

        Table<2, DoFTools::Coupling> preconditioner_coupling(dim + 1, dim + 1);

        for (unsigned int c = 0; c < dim + 1; ++c)
          for (unsigned int d = 0; d < dim + 1; ++d)
            if (((c == dim) && (d == dim)))
              preconditioner_coupling[c][d] = DoFTools::always;
            else
              preconditioner_coupling[c][d] = DoFTools::none;

        DoFTools::make_sparsity_pattern(dof_handler,
                                      preconditioner_coupling,
                                      preconditioner_dsp,
                                      constraints,
                                      false);

        preconditioner_sparsity_pattern.copy_from(preconditioner_dsp);
    }
    

    system_matrix.reinit(sparsity_pattern);
    stiff_preconditioner.reinit(preconditioner_sparsity_pattern);
    mass_preconditioner.reinit(preconditioner_sparsity_pattern);
    //mass_u_matrix.reinit(preconditioner_sparsity_pattern.block(0,0));
    stiff_p_matrix.reinit(preconditioner_sparsity_pattern.block(1,1));
    mass_p_matrix.reinit(preconditioner_sparsity_pattern.block(1,1));
    unconstrained_mass_matrix.reinit(unconstrained_sparsity_pattern);
    unconstrained_system_matrix.reinit(unconstrained_sparsity_pattern);

    mass_schur.reinit(preconditioner_sparsity_pattern.block(1,1));

    previous_solution.reinit(2);
    previous_solution.block(0).reinit(n_u);
    previous_solution.block(1).reinit(n_p);
    previous_solution.collect_sizes();

    current_solution.reinit(2);
    current_solution.block(0).reinit(n_u);
    current_solution.block(1).reinit(n_p);
    current_solution.collect_sizes();
    
    system_rhs.reinit(2);
    system_rhs.block(0).reinit(n_u);
    system_rhs.block(1).reinit(n_p);
    system_rhs.collect_sizes();
    
    load_vector.reinit(2);
    load_vector.block(0).reinit(n_u);
    load_vector.block(1).reinit(n_p);
    load_vector.collect_sizes();

    convective_term.reinit(2);
    convective_term.block(0).reinit(n_u);
    convective_term.block(1).reinit(n_p);
    convective_term.collect_sizes();

    // use exact solution to set up initial condition
    VectorTools::interpolate(dof_handler,
                             exact_solution,
                             current_solution);
    
}

template <int dim>
void NavierStokes<dim>::assemble_system()
{
    system_matrix         = 0;
    stiff_preconditioner = 0;
    mass_preconditioner = 0;
    //mass_u_matrix         = 0;
    stiff_p_matrix         = 0;
    mass_p_matrix         = 0;

    QGauss<dim> quadrature_formula(degree+2);

    MappingQGeneric<dim> mapping(1);
    FEValues<dim> fe_values(mapping,
                            fe,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                            update_gradients | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_system_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_stiff_preconditioner(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_mass_preconditioner(dofs_per_cell, dofs_per_cell);

    Tensor<1, dim>              phi_i_u, phi_j_u;
    Tensor<2, dim>              grad_phi_i_u, grad_phi_j_u;
    double                      phi_i_p, phi_j_p;
    Tensor<1, dim>              grad_phi_i_p, grad_phi_j_p;
    double                      div_phi_i_u, div_phi_j_u;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector u(0);
    const FEValuesExtractors::Scalar p(dim);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_mass_matrix = 0;
      cell_system_matrix = 0;
      cell_stiff_preconditioner = 0;
      cell_mass_preconditioner = 0;

      fe_values.reinit(cell);

      for (unsigned int q_index = 0; q_index < n_q_points;
             ++q_index)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              phi_i_u = fe_values[u].value(i, q_index);
              phi_i_p = fe_values[p].value(i, q_index);
              grad_phi_i_u = fe_values[u].gradient(i, q_index);
              grad_phi_i_p = fe_values[p].gradient(i, q_index);
              div_phi_i_u  = fe_values[u].divergence(i,q_index);
              

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  phi_j_u = fe_values[u].value(j, q_index);
                  phi_j_p = fe_values[p].value(j, q_index);
                  grad_phi_j_u = fe_values[u].gradient(j, q_index);
                  grad_phi_j_p = fe_values[p].gradient(j, q_index);
                  div_phi_j_u  = fe_values[u].divergence(j,q_index);

                  // mass matrix
                  cell_mass_matrix(i, j) +=
                      rho * phi_i_u * phi_j_u * fe_values.JxW(q_index);
                    //  ( phi_i_u * phi_j_u -
                    //    dt *
                    //        phi_i_u * ( phi_j_u * grad_phi_j_u ) )
                    //  * fe_values.JxW(q_index);
                  
                  // pressure mass matrix aka preconditioner
                  cell_mass_preconditioner(i, j) +=
                        phi_i_p * phi_j_p *
                        fe_values.JxW(q_index);

                  // pressure stiffness matrix aka preconditioner
                  cell_stiff_preconditioner(i, j) +=
                        grad_phi_i_p * grad_phi_j_p *
                        fe_values.JxW(q_index);

                  // system matrix
                  cell_system_matrix(i,j) +=
                       (mu * 
                        scalar_product(grad_phi_i_u, grad_phi_j_u) +
                       (eta + mu) * div_phi_i_u * div_phi_j_u -
                        div_phi_i_u * phi_j_p -
                        phi_i_p * div_phi_j_u ) * fe_values.JxW(q_index);

                }
            }
        }
      cell_system_matrix *= dt;
      cell_system_matrix.add(1.0, cell_mass_matrix);

      cell->get_dof_indices(local_dof_indices);

      constraints.distribute_local_to_global(
        cell_system_matrix, local_dof_indices, system_matrix);
      constraints.distribute_local_to_global(
        cell_stiff_preconditioner, local_dof_indices, stiff_preconditioner);
      constraints.distribute_local_to_global(
        cell_mass_preconditioner, local_dof_indices, mass_preconditioner);

      unconstrained_system_matrix.add(local_dof_indices, cell_system_matrix);
      unconstrained_mass_matrix.add(local_dof_indices, cell_mass_matrix);
    }
    //mass_u_matrix.copy_from(preconditioner_matrix.block(0,0));
    stiff_p_matrix.copy_from(stiff_preconditioner.block(1,1));
    mass_p_matrix.copy_from(mass_preconditioner.block(1,1));
}


template <int dim>
void NavierStokes<dim>::assemble_rhs()
{
    system_rhs = 0;

    QGauss<dim> quadrature_formula(degree+2);

    MappingQGeneric<dim> mapping(1);
    FEValues<dim> fe_values(mapping,
                            fe,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                            update_gradients | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<Tensor<1,dim>> previous_velocity_values(n_q_points);
    std::vector<Tensor<2,dim>> previous_velocity_gradients(n_q_points);

    Tensor<1, dim>              phi_i_u;
    Tensor<1, dim>              lorenz_force;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // get data for forcing function
    ForcingFunction<dim> forcing_function(eta, mu, mag_field);
    forcing_function.set_time(current_time + dt);
    std::vector<Vector<double>>  f_vals(n_q_points, Vector<double>(dim+1));

    const FEValuesExtractors::Vector u(0);
    const FEValuesExtractors::Scalar p(dim);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_rhs = 0;

      fe_values.reinit(cell);

      fe_values[u].get_function_values(previous_solution,
                                       previous_velocity_values);
      fe_values[u].get_function_gradients(previous_solution,
                                          previous_velocity_gradients);

      forcing_function.vector_value_list(fe_values.get_quadrature_points(), f_vals);

      for (unsigned int q_index = 0; q_index < n_q_points;
             ++q_index)
        {
          const Tensor<1,dim> x_q = fe_values.quadrature_point(q_index);
          lorenz_force = mag_field.get_force(x_q);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              phi_i_u = fe_values[u].value(i, q_index);
              const unsigned int component_i = 
                  fe.system_to_component_index(i).first;

              cell_rhs(i) += 
                  rho * phi_i_u * previous_velocity_values[q_index] +
                  dt * (
                  - rho *
                    previous_velocity_gradients[q_index] *
                    previous_velocity_values[q_index] * phi_i_u
                  //phi_i_u * (previous_velocity_values[q_index] *
                  //           previous_velocity_gradients[q_index])
                  + fe_values.shape_value(i,q_index) * 
                    f_vals[q_index](component_i)
                  //+ phi_i_u * f_vel +
                  //phi_i_p * f_vals[q_index](2)
                  + phi_i_u * lorenz_force 
                  ) * fe_values.JxW(q_index);

            }
        }

      cell->get_dof_indices(local_dof_indices);

      constraints.distribute_local_to_global(
        cell_rhs, local_dof_indices, system_rhs);

    }
}


template <int dim>
void NavierStokes<dim>::run()
{
    setup_grid();
    setup_dofs();

    std::cout << "   Number of active cells:       "
            << triangulation.n_active_cells() << std::endl;
    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << '\n'
            << "   Timestep size: "
            << dt
            << std::endl;

    assemble_system();

    // define block schur preconditioner
    BlockSchurPreconditioner block_preconditioner(
            eta, mu, dt, system_matrix, stiff_p_matrix, mass_p_matrix, mass_schur);

    ExactSolution<dim> exact_solution;

    for (; current_time <= 1; current_time += dt, ++time_step_number)
    {
      // The current solution should be swapped with the previous solution:
      std::swap(previous_solution, current_solution);


      // setup constraints for imposing boundary conditions
      {
        constraints.clear();
        exact_solution.set_time(current_time + dt);
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);

        //FEValuesExtractors::Scalar q(dim);

        // boundary condition.
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 0,
                                                 exact_solution,
                                                 constraints);
      }
      constraints.close();

      // set up constrained right hand side
      assemble_rhs();

      // solve the system using the block preconditioner
      SolverControl solver_control(system_matrix.m(), 1e-8*system_rhs.l2_norm(),true);
      SolverFGMRES<BlockVector<double>> gmres(solver_control);
      gmres.solve(system_matrix, current_solution, system_rhs, block_preconditioner);
      constraints.distribute(current_solution);

  }

  output_results();

}

template <int dim>
void NavierStokes<dim>::output_results() const
{
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
           dim, DataComponentInterpretation::component_is_part_of_vector); 
    data_component_interpretation.push_back(
           DataComponentInterpretation::component_is_scalar);

    // Compute the pointwise maximum error:
    Vector<double> max_error_per_cell(triangulation.n_active_cells());
    {
        ExactSolution<dim> exact_solution;
        exact_solution.set_time(current_time);

        MappingQGeneric<dim> mapping(1);
        VectorTools::integrate_difference(mapping,
                                          dof_handler,
                                          current_solution,
                                          exact_solution,
                                          max_error_per_cell,
                                          QIterated<dim>(QGauss<1>(2), 2),
                                          VectorTools::NormType::Linfty_norm);
        std::cout << "maximum error = "
                  << *std::max_element(max_error_per_cell.begin(),
                                       max_error_per_cell.end())
                  << std::endl;
    }

    // Save the output:
    {
        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(current_solution, 
                                 solution_names,
                                 DataOut<dim>::type_dof_data,
                                 data_component_interpretation);
        data_out.add_data_vector(max_error_per_cell, "max_error_per_cell");
        data_out.build_patches();

        std::ofstream output("output-" + std::to_string(dim) + "d-" +
                             std::to_string(n_global_refinements) + ".vtu");
        data_out.write_vtu(output);
    }
}


int
main()
{
  int degree = 1;

  for (unsigned int i = 0; i < 4; ++i)
    {
      NavierStokes<2> ns(degree, i);
      ns.run();
    }

  return 0;
}
